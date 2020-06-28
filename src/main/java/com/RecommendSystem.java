package com;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.recommendation.ALS;
import org.apache.spark.ml.recommendation.ALSModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;

import java.util.List;

public class RecommendSystem {
    public static void main(String[] args) {
        Logger.getLogger("org.apache").setLevel(Level.WARN);

        SparkSession spark = SparkSession.builder().appName("sparkML").master("local[*]").getOrCreate();

        Dataset<Row> csvData = spark.read()
                .option("header", true)
                .option("inferSchema", true)
                .csv("src/main/resources/VPPCourseViews.csv");

        csvData = csvData.withColumn("proportionWatched", functions.col("proportionWatched").multiply(100));
//        csvData.groupBy("userId").pivot("courseId").sum("proportionWatched").show();
//        csvData.show();

        Dataset<Row>[] trainAndHoldoutData = csvData.randomSplit(new double[]{0.9, 0.1});
        Dataset<Row> trainingData = trainAndHoldoutData[0];
        Dataset<Row> holdOutData = trainAndHoldoutData[1];

        ALS als = new ALS()
                .setMaxIter(10)
                .setRegParam(0.1)
                .setUserCol("userId")
                .setItemCol("courseId")
                .setRatingCol("proportionWatched");

        ALSModel model = als.fit(trainingData);
        Dataset<Row> userRecs = model.recommendForAllUsers(5);

        List<Row> userRecsList = userRecs.takeAsList(5);
        for (Row r : userRecsList) {
            int userId = r.getAs(0);
            String recs = r.getAs(1).toString();
            System.out.println("User " + userId + " we might want to recommend " + recs);
            System.out.println("This user has already watched: ");
            csvData.filter("userId = " + userId).show();
        }

    }
}

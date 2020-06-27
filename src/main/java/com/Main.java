package com;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.ml.feature.VectorAssembler;

public class Main {
    public static void main(String[] args) {
        Logger.getLogger("org.apache").setLevel(Level.WARN);
        SparkSession spark = SparkSession.builder().appName("sparkML").master("local[*]").getOrCreate();

        Dataset<Row> csvData = spark.read()
                .option("header", true)
                .option("inferSchema", true)
                .csv("src/main/resources/GymCompetition.csv");
//        csvData.show();
        csvData.printSchema();

        VectorAssembler vectorAssembler = new VectorAssembler();
        vectorAssembler.setInputCols(new String[]{"Age", "Height", "Weight"});
        vectorAssembler.setOutputCol("features");
        Dataset<Row> csvDataWithFeatures = vectorAssembler.transform(csvData);
//        csvDataWithFeatures.show();
        Dataset<Row> modelInputData = csvDataWithFeatures.select("NoOfReps", "features").withColumnRenamed("NoOfReps", "label");
//        modelInputData.show();

        LinearRegression linearRegression = new LinearRegression();
        LinearRegressionModel model = linearRegression.fit(modelInputData);
        System.out.println("bias: " + model.intercept());
        System.out.println("weight: " + model.coefficients());

        model.transform(modelInputData).show();
    }
}

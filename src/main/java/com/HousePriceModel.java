package com;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class HousePriceModel {
    public static void main(String[] args) {
        Logger.getLogger("org.apache").setLevel(Level.WARN);

        SparkSession spark = SparkSession.builder().appName("sparkML").master("local[*]").getOrCreate();

        Dataset<Row> csvData = spark.read()
                .option("header", true)
                .option("inferSchema", true)
                .csv("src/main/resources/kc_house_data.csv");
//        csvData.show();
//        csvData.printSchema();
        System.out.println(csvData.count());

        VectorAssembler vectorAssembler = new VectorAssembler();
        vectorAssembler.setInputCols(new String[]{"bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "grade"});
        vectorAssembler.setOutputCol("features");
        Dataset<Row> modelInputData = vectorAssembler.
                transform(csvData)
                .select("price", "features")
                .withColumnRenamed("price", "label");
//        modelInputData.show();
        Dataset<Row>[] trainingAndTestData = modelInputData.randomSplit(new double[]{0.8, 0.2});
        Dataset<Row> trainingData = trainingAndTestData[0];
        Dataset<Row> testData = trainingAndTestData[1];

        LinearRegression linearRegression = new LinearRegression();
        ParamGridBuilder paramGridBuilder = new ParamGridBuilder();
        ParamMap[] paramMap = paramGridBuilder.addGrid(
                linearRegression.regParam(),
                new double[]{0.01, 0.1, 0.5}).addGrid(linearRegression.elasticNetParam(),
                new double[]{0, 0.5, 1}
        ).build();

        System.out.println("r2 value: " + model.summary().r2());
        System.out.println("RMSE value: " + model.summary().rootMeanSquaredError());

        model.transform(testData).show();
        System.out.println("r2 value for test data: " + model.evaluate(testData).r2());
        System.out.println("RMSE value for test data: " + model.evaluate(testData).rootMeanSquaredError());
    }

}

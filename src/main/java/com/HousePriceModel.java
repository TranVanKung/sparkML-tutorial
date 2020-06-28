package com;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.tuning.TrainValidationSplit;
import org.apache.spark.ml.tuning.TrainValidationSplitModel;
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
        Dataset<Row>[] dataSplits = modelInputData.randomSplit(new double[]{0.8, 0.2});
        Dataset<Row> trainingAndTestData = dataSplits[0];
        Dataset<Row> holdOutData = dataSplits[1];

        LinearRegression linearRegression = new LinearRegression();
        ParamGridBuilder paramGridBuilder = new ParamGridBuilder();
        ParamMap[] paramMap = paramGridBuilder.addGrid(
                linearRegression.regParam(),
                new double[]{0.01, 0.1, 0.5}).addGrid(linearRegression.elasticNetParam(),
                new double[]{0, 0.5, 1}
        ).build();

        TrainValidationSplit trainValidationSplit = new TrainValidationSplit()
                .setEstimator(linearRegression)
                .setEvaluator(new RegressionEvaluator().setMetricName("r2"))
                .setEstimatorParamMaps(paramMap)
                .setTrainRatio(0.8);

        TrainValidationSplitModel model = trainValidationSplit.fit(trainingAndTestData);
        LinearRegressionModel lrModel = (LinearRegressionModel) model.bestModel();

        System.out.println("r2 value: " + lrModel.summary().r2());
        System.out.println("RMSE value: " + lrModel.summary().rootMeanSquaredError());

//        lrModel.transform(holdOutData).show();
        System.out.println("r2 value for test data: " + lrModel.evaluate(holdOutData).r2());
        System.out.println("RMSE value for test data: " + lrModel.evaluate(holdOutData).rootMeanSquaredError());
        System.out.println("weight: " + lrModel.coefficients() + " intercept: " + lrModel.intercept());
        System.out.println("regparam: " + lrModel.getRegParam() + " elastic net param: " + lrModel.getElasticNetParam());
    }

}

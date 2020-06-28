package com;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.IndexToString;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.DecisionTreeRegressionModel;
import org.apache.spark.ml.regression.DecisionTreeRegressor;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.functions;

import java.util.Arrays;
import java.util.List;

public class DecisionTreeModel {
    public static UDF1<String, String> countryGrouping = new UDF1<String, String>() {
        public String call(String country) throws Exception {
            List<String> topCountries = Arrays.asList(new String[]{"GB", "US", "IN", "UNKNOWN"});
            List<String> europeanCountries = Arrays.asList(new String[]{"BE", "BG", "CZ", "DK", "DE", "EE", "IE", "EL", "ES", "FR", "HR", "IT", "CY", "LV", "LT", "LU", "HU", "MT", "NL", "AT", "PL", "PT", "RO", "SI", "SK", "FI", "SE", "CH", "IS", "NO", "LI", "EU"});

            if (topCountries.contains(country)) return country;
            if (europeanCountries.contains(country)) return "EUROPE";
            else return "OTHER";
        }
    };

    public static void main(String[] args) {
        Logger.getLogger("org.apache").setLevel(Level.WARN);

        SparkSession spark = SparkSession.builder().appName("sparkML").master("local[*]").getOrCreate();

        Dataset<Row> csvData = spark.read()
                .option("header", true)
                .option("inferSchema", true)
                .csv("src/main/resources/vppFreeTrials.csv");
        spark.udf().register("countryGrouping", countryGrouping, DataTypes.StringType);
        csvData = csvData
                .withColumn("country", functions.callUDF("countryGrouping", functions.col("country")))
                .withColumn("label", functions.when(functions.col("payments_made").geq(1), functions.lit(1)).otherwise(functions.lit(0)));

        StringIndexer countryIndexer = new StringIndexer();
        csvData = countryIndexer
                .setInputCol("country")
                .setOutputCol("countryIndex")
                .fit(csvData)
                .transform(csvData);

        Dataset<Row> countryIndexes = csvData.select("countryIndex").distinct();
        IndexToString indexToString = new IndexToString();
        indexToString.setInputCol("countryIndex").setOutputCol("value").transform(countryIndexes).show();

        VectorAssembler vectorAssembler = new VectorAssembler();
        vectorAssembler.setInputCols(new String[]{"countryIndex", "rebill_period", "chapter_access_count", "seconds_watched"});
        vectorAssembler.setOutputCol("features");
        Dataset<Row> inputData = vectorAssembler.transform(csvData).select("label", "features");
        inputData.show();

        Dataset<Row>[] trainAndHoldoutData = inputData.randomSplit(new double[]{0.9, 0.1});
        Dataset<Row> trainingData = trainAndHoldoutData[0];
        Dataset<Row> holdOutData = trainAndHoldoutData[1];

        DecisionTreeClassifier dtClassifier = new DecisionTreeClassifier();
        dtClassifier.setMaxDepth(3);
        DecisionTreeClassificationModel model = dtClassifier.fit(trainingData);

        Dataset<Row> predictions = model.transform(holdOutData);

//        predictions.show();
//        System.out.println(model.toDebugString());

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator();
        evaluator.setMetricName("accuracy");
        System.out.println("accuracy of decision tree is: " + evaluator.evaluate(predictions));

        RandomForestClassifier rfClassifier = new RandomForestClassifier();
        rfClassifier.setMaxDepth(3);
        RandomForestClassificationModel rfModel = rfClassifier.fit(trainingData);
        Dataset<Row> predictions2 = rfModel.transform(holdOutData);
        predictions2.show();

        System.out.println(rfModel.toDebugString());
        System.out.println("accuracy of random forest is : " + evaluator.evaluate(predictions2));
    }
}

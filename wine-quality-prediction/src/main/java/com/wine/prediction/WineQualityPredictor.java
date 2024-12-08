package com.wine.prediction;

import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.DataTypes;
import java.io.IOException;

public class WineQualityPredictor {
    private static final String MASTER_URL = "spark://54.173.237.74:7077";
    private static final String TRAINING_DATA_PATH = "data/TrainingDataset.csv";
    private static final String VALIDATION_DATA_PATH = "data/ValidationDataset.csv";
    private static final String MODEL_PATH = "models/wine-quality-model";
    
    public static void main(String[] args) {
        if (args.length != 1) {
            System.out.println("Usage: WineQualityPredictor <test-file>");
            System.exit(1);
        }

        SparkSession spark = SparkSession.builder()
            .appName("Wine Quality Prediction")
            .master(MASTER_URL)
            .getOrCreate();

        try {
            // Define schema for CSV
            StructType schema = new StructType(new StructField[]{
                DataTypes.createStructField("fixed_acidity", DataTypes.DoubleType, true),
                DataTypes.createStructField("volatile_acidity", DataTypes.DoubleType, true),
                DataTypes.createStructField("citric_acid", DataTypes.DoubleType, true),
                DataTypes.createStructField("residual_sugar", DataTypes.DoubleType, true),
                DataTypes.createStructField("chlorides", DataTypes.DoubleType, true),
                DataTypes.createStructField("free_sulfur_dioxide", DataTypes.DoubleType, true),
                DataTypes.createStructField("total_sulfur_dioxide", DataTypes.DoubleType, true),
                DataTypes.createStructField("density", DataTypes.DoubleType, true),
                DataTypes.createStructField("pH", DataTypes.DoubleType, true),
                DataTypes.createStructField("sulphates", DataTypes.DoubleType, true),
                DataTypes.createStructField("alcohol", DataTypes.DoubleType, true),
                DataTypes.createStructField("quality", DataTypes.DoubleType, true)
            });

            // Load training data
            Dataset<Row> training = spark.read()
                .option("header", "true")
                .option("delimiter", ";")
                .schema(schema)
                .csv(TRAINING_DATA_PATH);

            // Load validation data
            Dataset<Row> validation = spark.read()
                .option("header", "true")
                .option("delimiter", ";")
                .schema(schema)
                .csv(VALIDATION_DATA_PATH);

            // Feature columns
            String[] featureColumns = new String[]{
                "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
                "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
                "pH", "sulphates", "alcohol"
            };

            // Create pipeline
            VectorAssembler assembler = new VectorAssembler()
                .setInputCols(featureColumns)
                .setOutputCol("features");

            RandomForestClassifier rf = new RandomForestClassifier()
                .setLabelCol("quality")
                .setFeaturesCol("features")
                .setNumTrees(100)
                .setMaxDepth(5);

            Pipeline pipeline = new Pipeline()
                .setStages(new org.apache.spark.ml.PipelineStage[]{
                    assembler, rf
                });

            // Train model
            System.out.println("Training model...");
            PipelineModel model = pipeline.fit(training);

            // Make predictions on validation data
            System.out.println("Making predictions on validation data...");
            Dataset<Row> predictions = model.transform(validation);

            // Evaluate model
            MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("quality")
                .setPredictionCol("prediction")
                .setMetricName("f1");

            double f1Score = evaluator.evaluate(predictions);
            System.out.println("F1 Score on validation data: " + f1Score);

            // Save the model
            try {
                System.out.println("Saving model...");
                model.save(MODEL_PATH);
            } catch (IOException e) {
                System.err.println("Error saving model: " + e.getMessage());
                e.printStackTrace();
            }

            // Test file prediction
            System.out.println("Making predictions on test data...");
            Dataset<Row> testData = spark.read()
                .option("header", "true")
                .option("delimiter", ";")
                .schema(schema)
                .csv(args[0]);

            Dataset<Row> testPredictions = model.transform(testData);
            double testF1Score = evaluator.evaluate(testPredictions);
            System.out.println("F1 Score on test data: " + testF1Score);

            // Display predictions
            System.out.println("\nSample Predictions:");
            testPredictions.select("quality", "prediction").show(5);

        } finally {
            spark.stop();
        }
    }
}

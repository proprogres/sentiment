package pl.org.proprogres.ml.sentiment;


import org.apache.spark.ml.Pipeline;

import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.NaiveBayes;  
import org.apache.spark.ml.classification.NaiveBayesModel;
import org.apache.spark.ml.feature.RegexTokenizer;
import org.apache.spark.ml.feature.StopWordsRemover;
import org.apache.spark.ml.feature.NGram;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.IndexToString;
import org.apache.spark.ml.feature.Tokenizer;

//import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}  
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;  
import org.apache.spark.mllib.util.MLUtils;

import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;

import static org.apache.spark.sql.functions.col;


import org.apache.spark.sql.Row;
import org.apache.spark.sql.Dataset;  
import org.apache.spark.storage.StorageLevel;
import org.apache.spark.sql.SQLContext;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;

import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.feature.HashingTF;

import pl.org.proprogres.ml.sentiment.transformer.Stemmer;

import org.apache.spark.sql.types.IntegerType;


public class Amazon {
	public static void main(String[] args) {
		SparkConf conf = new SparkConf().setAppName("Simple Application");
    JavaSparkContext sc = new JavaSparkContext(conf);
    sc.setLogLevel("ERROR");
    SQLContext sqlContext = new SQLContext(sc);
	    
		String reviews = "/home/la/training_twitter_by_smiles_287000.csv";
		
		Dataset<Row> reviewsDs = sqlContext.read()
      .option("sep", "|")
      .option("header", true)
      .csv(reviews);
    
		reviewsDs.registerTempTable("reviews");
		
		Dataset<Row> reviewsDF = sqlContext.sql("SELECT SENTIMENT as label, TEXT as text FROM reviews WHERE SENTIMENT = -1 OR SENTIMENT = 1 ORDER BY rand()");
    reviewsDF = reviewsDF
      .withColumn("label", col("label").cast("Double"))
      //.withColumn("text", col("text"))
      .select("label", "text");
		reviewsDF.persist(org.apache.spark.storage.StorageLevel.MEMORY_AND_DISK());
		
		System.out.println(reviewsDF.count());
		System.out.println("--------------------------------");
		
		//reviewsDF.show(100);
		
		Tokenizer tokenizer = new Tokenizer()
				  .setInputCol("text")
				  .setOutputCol("words");
    Dataset<Row> wordsDF = tokenizer.transform(reviewsDF);
		
		StopWordsRemover remover = new StopWordsRemover()
				  .setInputCol(tokenizer.getOutputCol())
				  .setOutputCol("filtered");
		Dataset<Row> filteredDF = remover.transform(wordsDF);
    
    NGram ngram = new NGram()
		  .setInputCol(remover.getOutputCol())
      .setOutputCol("ngrammed")
		  .setN(1);
    Dataset<Row> ngramDF = ngram.transform(filteredDF);
		//ngramDF.show(50);
    
		HashingTF hasher = new HashingTF()
      .setNumFeatures(60000)
      .setInputCol(ngram.getOutputCol())
      .setOutputCol("features");
    Dataset<Row> featuresDF = hasher.transform(ngramDF);
    // System.out.println("Number of features: " + hasher.numFeatures());
      
    NaiveBayes nb = new NaiveBayes();
		  
		Pipeline pipeline = new Pipeline()
      .setStages(new PipelineStage[] {tokenizer, remover, ngram, hasher, nb});
		  
		
		Dataset<Row>[] splits = reviewsDF.randomSplit(new double[]{0.6, 0.4});
    Dataset<Row> training = splits[0];
    Dataset<Row> test = splits[1];
    //training.show(10);
    //test.show(10);
    // Dataset<Row> training = reviewsDF.filter(col("rowNumber").lt(3000)); 
    // Dataset<Row> test = reviewsDF.filter(col("rowNumber").gt(3000));

    
    PipelineModel model = pipeline.fit(training);
    
    Dataset<Row> predictions = model.transform(test);
    System.out.println(model.toString());
    
    //predictions.select("prediction", "label", "features").show(200);
    MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy");
    double accuracy = evaluator.evaluate(predictions);
    System.out.println("Test Error = " + (1.0 - accuracy));
    
    // Select (prediction, true label) and compute test error.
    /*MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy");
    double accuracy = evaluator.evaluate(predictions);
    System.out.println("Test Error = " + (1.0 - accuracy));

    DecisionTreeClassificationModel treeModel =
      (DecisionTreeClassificationModel) (model.stages()[2]);
    System.out.println("Learned classification tree model:\n" + treeModel.toDebugString());
    */
    
		/* 
		Tokenizer tokenizer = new Tokenizer()
		  .setInputCol("text")
		  .setOutputCol("words");
		
		StopWordsRemover remover = new StopWordsRemover()
		  .setInputCol(tokenizer.getOutputCol())
		  .setOutputCol("filtered");
		  
		NGram ngram3 = new NGram()
		  .setInputCol(remover.getOutputCol())
		  .setN(3);
		  
		HashingTF hashingTF = new HashingTF()
      .setNumFeatures(100)
      .setInputCol(ngram3.getOutputCol())
      .setOutputCol("features");
      
    //Dataset<Row> tokenized = hashingTF.transform(reviewsDF);
    //tokenized.select("features").show(false);
      
    LogisticRegression lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.01);
      
    NaiveBayes nb = new NaiveBayes();
		  
		Pipeline pipeline = new Pipeline()
      .setStages(new PipelineStage[] {tokenizer, remover, ngram3, hashingTF, nb});
		  
    Dataset<Row> training = reviewsDF.filter(col("rowNumber").lt(1000)); 
    Dataset<Row> test = reviewsDF.filter(col("rowNumber").gt(1000));

    
    PipelineModel model = pipeline.fit(training);*/
    
    /* Dataset<Row> predictions = model.transform(test);
    for (Row r : predictions.select("probability", "prediction").collectAsList()) {
      System.out.println("prob=" + r.get(0) + ", prediction=" + r.get(1));
    } */
    
    //System.out.println("---------------------------------------------------");
    //System.out.println("Training data: " + training.count());
    //System.out.println("Test data: " + test.count());
    
	}
}

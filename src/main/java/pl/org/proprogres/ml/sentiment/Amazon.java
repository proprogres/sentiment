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

import static org.apache.spark.sql.functions.col;


import org.apache.spark.sql.Row;
import org.apache.spark.sql.Dataset;  
import org.apache.spark.storage.StorageLevel;
import org.apache.spark.sql.SQLContext;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;

import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.feature.HashingTF;


public class Amazon {
	public static void main(String[] args) {
		SparkConf conf = new SparkConf().setAppName("Simple Application");
	    JavaSparkContext sc = new JavaSparkContext(conf);
	    sc.setLogLevel("ERROR");
	    SQLContext sqlContext = new SQLContext(sc);
	    
		String reviews = "/home/la/Downloads/Amazon_Instant_Video_5.json";
		
		Dataset<Row> reviewsDs = sqlContext.read().json(reviews);
		reviewsDs.registerTempTable("reviews");
	
		String query = new StringBuilder()
				.append("SELECT text, label, rowNumber FROM (SELECT ")
				.append("reviews.overall AS label ,reviews.reviewText AS text ,row_number() OVER (PARTITION BY overall ORDER BY rand()) AS rowNumber")
				.append(" FROM reviews) reviews WHERE rowNumber <= 1700")
				.toString();
		
		Dataset<Row> reviewsDF = sqlContext.sql(query);
		reviewsDF.persist(org.apache.spark.storage.StorageLevel.MEMORY_AND_DISK());
		
		reviewsDF.groupBy("label").count().orderBy("label").show();
		
		Tokenizer tokenizer = new Tokenizer()
				  .setInputCol("text")
				  .setOutputCol("words");
		
		Dataset<Row> tokenizedDF = tokenizer.transform(reviewsDF);
		// tokenizedDF.show();
		
		StopWordsRemover remover = new StopWordsRemover()
				  .setInputCol(tokenizer.getOutputCol())
				  .setOutputCol("filtered");
		
		Dataset<Row> stopwordsRemovedDF = remover.transform(tokenizedDF);
		stopwordsRemovedDF.show();
		
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

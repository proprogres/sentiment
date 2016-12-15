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


import org.apache.spark.sql.Row;
import org.apache.spark.sql.Dataset;  
import org.apache.spark.storage.StorageLevel;
import org.apache.spark.sql.SQLContext;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;


public class Amazon {
	public static void main(String[] args) {
		SparkConf conf = new SparkConf().setAppName("Simple Application");
	    JavaSparkContext sc = new JavaSparkContext(conf);
	    SQLContext sqlContext = new SQLContext(sc);
	    
		String reviews = "/home/la/Amazon_Instant_Video_5.json";
		
		Dataset<Row> reviewsDs = sqlContext.read().json(reviews);
		reviewsDs.registerTempTable("reviews");
	
		String query = new StringBuilder()
				.append("SELECT text, label, rowNumber FROM (SELECT ")
				.append("reviews.overall AS label ,reviews.reviewText AS text ,row_number() OVER (PARTITION BY overall ORDER BY rand()) AS rowNumber")
				.append(" FROM reviews) reviews WHERE rowNumber <= 1000")
				.toString();
		
		Dataset<Row> reviewsDF = sqlContext.sql(query);
		reviewsDF.persist(org.apache.spark.storage.StorageLevel.MEMORY_AND_DISK());
		
		reviewsDF.groupBy("label").count().orderBy("label").show();
		
		Tokenizer tokenizer = new Tokenizer()
		  .setInputCol("text")
		  .setOutputCol("words");
		
		StopWordsRemover remover = new StopWordsRemover()
		  .setInputCol(tokenizer.getOutputCol())
		  .setOutputCol("filtered");
		  
		NGram ngram3 = new NGram()
		  .setInputCol(remover.getOutputCol())
		  .setN(3);
		  
		Pipeline pipeline = new Pipeline()
      .setStages(new PipelineStage[] {tokenizer, remover, ngram3});
		  
    Dataset<Row> training = reviewsDF.filter(reviewsDF("rowNumber") <= 15000).select("text","label"); 
    Dataset<Row> test = reviewsDF.filter(reviewsDF("rowNumber") > 15000).select("text","label");

    
    PipelineModel model = pipeline.fit(training);
    
    Dataset<Row> predictions = model.transform(test);
    for (Row r : predictions.select("id", "text", "probability", "prediction").collectAsList()) {
      System.out.println("(" + r.get(0) + ", " + r.get(1) + ") --> prob=" + r.get(2)
        + ", prediction=" + r.get(3));
    }
    
	}
}

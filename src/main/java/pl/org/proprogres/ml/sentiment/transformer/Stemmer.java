package pl.org.proprogres.ml.sentiment.transformer;

import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.types.StructType;

public class Stemmer extends Transformer {
	private static final long serialVersionUID = 5545470640951989469L;
    String column;
    int power = 1;

	Stemmer(String column, int power) {
	    this.column = column;
	    this.power = power;
	}
	
	
	@Override
	public String uid() {
	   return "CustomTransformer" + serialVersionUID;
	}
	
	@Override
	public Transformer copy(ParamMap arg0) {
	   return null;
	}
	
	@Override
	public Dataset transform(Dataset<?> data) {
	   return data.withColumn("power", functions.pow(data.col(this.column), this.power));
	}
	
	@Override
	public StructType transformSchema(StructType arg0) {
	   return arg0;
	}
}

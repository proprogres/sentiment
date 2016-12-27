package pl.org.proprogres.ml.sentiment.transformer;

import org.apache.spark.ml.UnaryTransformer;
import org.apache.spark.sql.types.ArrayType;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StringType;
import scala.Function1;
import java.util.List;
import java.io.Serializable;
import org.apache.spark.ml.feature.Tokenizer;

public class Stemmer extends UnaryTransformer<String, List<String>, Tokenizer> implements Serializable {

    private final static String uid = Tokenizer.class.getSimpleName() + "_" + UUID.randomUUID().toString();

    private static Map<String, String> stringReplaceMap;

    @Override
    public void validateInputType(DataType inputType) {
        assert (inputType.equals(DataTypes.StringType)) :
                String.format("Input type must be %s, but got %s", DataTypes.StringType.simpleString(), inputType.simpleString());
    }

    public Function1<String, List<String>> createTransformFunc() {
        Function1<String, List<String>> f = new TokenizerFunction();
        return f;
    }

    public DataType outputDataType() {
        return DataTypes.createArrayType(DataTypes.StringType, true);
    }

    public String uid() {
        return uid;
    }

    private class TokenizerFunction extends AbstractFunction1<String, List<String>> implements Serializable {
        public List<String> apply(String sentence) {
             ... code goes here
        }
    }

}

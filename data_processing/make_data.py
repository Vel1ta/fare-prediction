from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, DoubleType
import pyspark

def one_hot_encode(df, variables):
    # Create a copy of the original DataFrame
    encoded_df = df
    
    # Iterate over the variables and perform one-hot encoding
    for variable in variables:
        # Create a unique column name for the encoded variable
        encoded_column = variable + "_encoded"
        
        # StringIndexer to convert the categorical variable into numerical index
        indexer = StringIndexer(inputCol=variable, outputCol=encoded_column)
        indexed_df = indexer.fit(encoded_df).transform(encoded_df)
        
        # OneHotEncoder to perform one-hot encoding
        encoder = OneHotEncoder(inputCols=[encoded_column], outputCols=[encoded_column + "_onehot"])
        encoded_df = encoder.fit(indexed_df).transform(indexed_df)
        
        # Drop the original indexed column
        encoded_df = encoded_df.drop(encoded_column)
    
    # Return the encoded DataFrame
    return encoded_df

def main():
    print(pyspark.__version__)
    data = spark.read.table("samples.nyctaxi.trips")
    categoricalColumns = ["pickup_zip","dropoff_zip"]
    
    data_vector = one_hot_encode(data, categoricalColumns)
    print(data_vector)

# Main execution
if __name__=="__main__":
    main()
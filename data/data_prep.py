# Basic Python Libraries
import yaml

# Query Functions Libraries
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
import pyspark

# Data Bricks Features Libraries
import uuid 
#from databricks.feature_store.client import FeatureStoreClient
#from databricks.feature_store.entities.feature_lookup import FeatureLookup

def get_data_id(df, id_name):
    df_with_id = df.withColumn(id_name, F.monotonically_increasing_id())

    return df_with_id

def get_data_df(df, datetime_column):
    for col in datetime_column:
        df = df.withColumn("datetime", 
                            F.to_timestamp(df[col], 
                            "yyyy-MM-dd HH:mm:ss"))

        df = df.withColumn(col + "_year", F.year("datetime")) \
               .withColumn(col + "_month", F.month("datetime")) \
               .withColumn(col + "_day", F.dayofmonth("datetime")) \
               .withColumn(col + "_hour", F.hour("datetime"))
        
    #df = df.drop("datetime")
    return df

def get_encode_df(df, columns_to_encode):
    spark = SparkSession.builder.getOrCreate()
    temp_view_name = "temp_view"

    # Register DataFrame as a temporary view
    df.createOrReplaceTempView(temp_view_name)

    # Generate the SQL expression for one-hot encoding
    sql_expression = ", ".join(
        [f"CASE WHEN {col_name} = '{col_value}' THEN 1 ELSE 0 END AS {col_name}_{col_value}"
         for col_name in columns_to_encode
         for col_value in df.select(F.col(col_name)).distinct().rdd.flatMap(lambda x: x).collect()]
         )

    # Perform one-hot encoding using SQL expression
    encoded_df = spark.sql(
        f"SELECT *, {sql_expression} FROM {temp_view_name}"
        )
    encoded_df = encoded_df.drop(columns_to_encode)

    return encoded_df


def main():
    # Read Config
    with open('../config.yaml') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    raw_data_name = config['raw_data_name']
    processed_data_name = config['processed_data_name']
    database_name = config['database_name']
    model_name = config['model_name']

    # Name Data
    run_id = str(uuid.uuid4()).replace('-', '')
 
    database_name = f"fare_prediction_{run_id}"
    model_name = f"pit_demo_model_{run_id}"
    
    # Create the database
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {database_name}")

    data = spark.read.table(raw_data_name)
    dateColumns = ["tpep_pickup_datetime", "tpep_dropoff_datetime"]
    categoricalColumns = dateColumns + ["pickup_zip", "dropoff_zip"]

    data = get_data_id(data, "trip_id")
    data = get_data_df(data, dateColumns)

    

    
    

# Main execution
if __name__=="__main__":
    main()
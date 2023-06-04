from delta.tables import *

input_delta_table = DeltaTable.forName(spark, "fareprediction.train")

# The Delta Live Tables table name for the input table that will be used in the pipeline code below.
input_dlt_table_name = "train"

# The input table schema stored as an array of strings. This is used to pass in the schema to the model predict udf.
input_dlt_table_columns = input_delta_table.toDF().columns

import mlflow

model_uri = f"models:/fareprediction/4"

# create spark user-defined function for model prediction.
# Note: : Here we use virtualenv to restore the python environment 
# that was used to train the model.
predict = mlflow.pyfunc.spark_udf(spark, 
                                  model_uri, 
                                  result_type="double", 
                                  env_manager='virtualenv')

import dlt
from pyspark.sql.functions import struct

@dlt.table(
  comment="DLT for predictions scored by fareprediction model based on fareprediction.train Delta table.",
  table_properties={
    "quality": "gold"
  }
)
def fareprediction_predictions():
  return (
    dlt.read(input_dlt_table_name)
    .withColumn('prediction', predict(struct(input_dlt_table_columns)))
  )

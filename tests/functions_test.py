from pyspark.sql import SparkSession
import pytest
import numpy as np

@pytest.fixture
def spark() -> SparkSession:
  # Create a SparkSession (the entry point to Spark functionality) on
  # the cluster in the remote Databricks workspace. Unit tests do not
  # have access to this SparkSession by default.
  return SparkSession.builder.getOrCreate()

def test_positive_column_train(spark):
  data_train = spark.read.table(f"fareprediction.train") 
  trip_distance = data_train.select("trip_distance").toPandas()
  min_v = trip_distance.trip_distance.min()
  assert min_v >= 0
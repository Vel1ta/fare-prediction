from pyspark.sql import SparkSession
import pytest

@pytest.fixture
def spark() -> SparkSession:
  # Create a SparkSession (the entry point to Spark functionality) on
  # the cluster in the remote Databricks workspace. Unit tests do not
  # have access to this SparkSession by default.
  return SparkSession.builder.getOrCreate()

def test_positive_column_train(spark):
  spark.sql(f'USE hive_metastore.fareprediction')
  min = spark.sql(f'SELECT MIN(fare_amount) FROM train')
  assert min > 0
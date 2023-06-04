import pytest

def test_positive_column_train(df_name, column_name):
  spark.sql(f'USE {df_name}')
  min = spark.sql(f'SELECT MIN({column_name}) FROM train')
  assert min > 0
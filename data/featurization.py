# Basic Python Libraries
import yaml

# Query Functions Libraries
from pyspark.sql import functions as F

# Data Prep
from data_prep import get_data_id
from data_prep import get_data_df


# Data Bricks Features Libraries
import uuid 
from databricks.feature_store.client import FeatureStoreClient
from databricks.feature_store.entities.feature_lookup import FeatureLookup


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

    data = data.withColumn("trip_time",
                F.expr("(datediff(tpep_dropoff_datetime, tpep_pickup_datetime) * 24 * 60) " \
                       "+ (hour(tpep_dropoff_datetime) * 60 + minute(tpep_dropoff_datetime)) " \
                        "- (hour(tpep_pickup_datetime) * 60 + minute(tpep_pickup_datetime))"))

    data_core = data.select(
                    "datetime",
                    "trip_id",
                    "fare_amount",
                    "trip_distance",
                    "trip_time"
    )
    
    data_dropoff = data.select(
                    "datetime",
                    "trip_id",
                    "dropoff_zip",
                    "tpep_dropoff_datetime_year",
                    "tpep_dropoff_datetime_month",
                    "tpep_dropoff_datetime_day",
                    "tpep_dropoff_datetime_hour",
    )

    data_pickup = data.select(
                    "datetime",
                    "trip_id",
                    "pickup_zip",
                    "tpep_pickup_datetime_year",
                    "tpep_pickup_datetime_month",
                    "tpep_pickup_datetime_day",
                    "tpep_pickup_datetime_hour",
    )


    fs = FeatureStoreClient()
    
    fs.create_table(
        f"{database_name}.core",
        primary_keys=["trip_id"],
        timestamp_keys=["datetime"],
        df=data_core,
        description="Variables unrelated to dates",
    ) 

    fs.create_table(
        f"{database_name}.dropoff",
        primary_keys=["trip_id"],
        timestamp_keys=["datetime"],
        df=data_dropoff,
        description="Variables unrelated to dates",
    ) 

    fs.create_table(
        f"{database_name}.pickup",
        primary_keys=["trip_id"],
        timestamp_keys=["datetime"],
        df=data_pickup,
        description="Variables unrelated to dates",
    ) 


# Main execution
if __name__=="__main__":
    main()
## Basic Libraries
import mlflow
import yaml

## Our data libraries
from data.featurization import get_data
from pyspark.sql.functions import struct, col

def main():
    output_table_path = "/FileStore/batch-inference/fareprediction"

    with open('../config.yaml') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    EXPERIMENT_ID = dict(
            mlflow.get_experiment_by_name(
                f'/Users/sveland1@eafit.edu.co/{config["model_name"]}-hyperop')
                )['experiment_id']
    
    runs_df = mlflow.search_runs(
                        experiment_ids=EXPERIMENT_ID, 
                        order_by=['metrics.mean_squared_error ASC']
                        )
    
    logged_model = f"runs:/{runs_df['run_id'].iloc[0]}/model"

    # Load Model
    loaded_model = mlflow.pyfunc.spark_udf(spark, 
                                           model_uri=logged_model, 
                                           result_type='double')
    core_data = spark.read.table(f"{config['model_name']}.core") 
    data_train = get_data(config, core_data, True)

    # Results inference
    results = data_train.withColumn('predictions', 
                          loaded_model(struct(*map(col, data_train.columns))))
    
    from datetime import datetime

    # To write to a unity catalog table, see instructions above
    results.write.save(f"{output_table_path}_{datetime.now().isoformat()}".replace(":", "."))
    
    print(results.show())



if __name__=="__main__":
    main()


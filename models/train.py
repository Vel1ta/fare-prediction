# Basic DS Libraries
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.ensemble
from sklearn.metrics import mean_squared_error
import yaml

# Hyperparam
from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK
from hyperopt.pyll import scope


# MLflow libraries
from math import exp
import mlflow.xgboost
from mlflow.models.signature import infer_signature
import numpy as np
import xgboost as xgb

# Databricks Libraries
from databricks.feature_store.client import FeatureStoreClient
from databricks.feature_store.entities.feature_lookup import FeatureLookup 
from data.featurization import get_data

search_space = {
  'n_estimators': scope.int(hp.quniform('n_estimators', 20, 1000, 1)),
  'learning_rate': hp.loguniform('learning_rate', -3, 0),
  'max_depth': scope.int(hp.quniform('max_depth', 2, 5, 1)),
}

def main():
    with open('../config.yaml') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    core_data = spark.read.table(f"{config['model_name']}.core") 
    data_train = get_data(config, core_data, True)
    data_train = data_train.select("*").toPandas()
    
    try:
        EXPERIMENT_ID = mlflow.create_experiment(
                            f"/Users/sveland1@eafit.edu.co/{config['model_name']}-hyperop"
                            )
    except:
        EXPERIMENT_ID = dict(mlflow.get_experiment_by_name(
                            f"/Users/sveland1@eafit.edu.co/{config['model_name']}-hyperop")
                            )['experiment_id']
    
    
    # train
    X_train, X_rem, y_train, y_rem = sklearn.model_selection.train_test_split(
        data_train.drop(["fare_amount"], axis=1),
        data_train["fare_amount"],
        test_size=0.4,
        random_state=1
    )

    # Split the remaining data equally into validation and test
    X_val, X_test, y_val, y_test = sklearn.model_selection.train_test_split(
    X_rem, 
    y_rem, 
    test_size=0.5, 
    random_state=123
    )

    def train_model(params):
            # With MLflow autologging, hyperparameters and the trained model are automatically logged to MLflow.
        mlflow.xgboost.autolog()
        with mlflow.start_run(experiment_id=EXPERIMENT_ID, nested=True):
            train = xgb.DMatrix(data=X_train, label=y_train)
            validation = xgb.DMatrix(data=X_val, label=y_val)
            # Pass in the validation set so xgb can track an evaluation metric.
            booster = xgb.train(params=params, dtrain=train, num_boost_round=1000,\
                                evals=[(validation, "validation")], early_stopping_rounds=50)
            validation_predictions = booster.predict(validation)
            mse = mean_squared_error(y_val, validation_predictions)
            mlflow.log_metric('mean_squared_error', mse)
        
            signature = infer_signature(X_train, booster.predict(train))
            mlflow.xgboost.log_model(booster, "model", signature=signature)
            
            
        return {'status': STATUS_OK, 'loss': mse, 'booster': booster.attributes()}

    spark_trials = SparkTrials(parallelism=10)

    with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name='xgboost_experiments'):
        best_params = fmin(
            fn=train_model, 
            space=search_space, 
            algo=tpe.suggest, 
            max_evals=96,
            trials=spark_trials,
        )


# Main execution
if __name__=="__main__":
    main()
## Basic Libraries
import mlflow
import yaml

def main():
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
    
    best_run = runs_df.iloc[0]
    best_artifact_uri = best_run['artifact_uri']

    # Load model
    print(best_artifact_uri)

    # Register model
    result = mlflow.register_model(
        best_artifact_uri+'/model', config["model_name"]
    )

# Main execution
if __name__=="__main__":
    main()
import sys
import house_price_prediction.data_scripts.split_data as sd
import house_price_prediction.data_scripts.create_features as cf
import house_price_prediction.model_training_scripts.train as tr

import mlflow

with mlflow.start_run(run_name='DATA_PROCESS_RUN') as parent_run:
    remote_server_uri = "http://0.0.0.0:5000"
    mlflow.set_tracking_uri(remote_server_uri)
    exp_name = "House_price_prediction_data_processing"
    mlflow.set_experiment(exp_name)
    mlflow.log_param("parent", "yes")
    sd.main()
    cf.main()
    tr.main()

    query = "tags.mlflow.parentRunId = '{}'".format(parent_run.info.run_id)
    results = mlflow.search_runs(filter_string=query)
    print(results[["run_id", "params.child", "tags.mlflow.runName"]])

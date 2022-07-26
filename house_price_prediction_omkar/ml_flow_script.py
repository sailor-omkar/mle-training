import sys
from distutils.log import debug

import mlflow
from flask import Flask


import house_price_prediction_omkar.data_scripts.create_features as cf
import house_price_prediction_omkar.data_scripts.split_data as sd
import house_price_prediction_omkar.model_training_scripts.train as tr
import house_price_prediction_omkar.data_scripts.download_data as dd

app = Flask(__name__)


@app.route("/run_ml_pipeline")
def run_ml_project():
    with mlflow.start_run(run_name="DATA_PROCESS_RUN") as parent_run:
        remote_server_uri = "http://0.0.0.0:5000"
        mlflow.set_tracking_uri(remote_server_uri)
        exp_name = "House_price_prediction_data_processing"
        mlflow.set_experiment(exp_name)
        mlflow.log_param("parent", "yes")
        dd.main()
        print('Download data.....')
        sd.main()
        print("Splitted data.....")
        cf.main()
        print("Created the feature.....")
        tr.main()
        print("Training of models done....")
        query = "tags.mlflow.parentRunId = '{}'".format(parent_run.info.run_id)
        results = mlflow.search_runs(filter_string=query)
        print(results[["run_id", "params.child", "tags.mlflow.runName"]])
        html_results = results.to_html()
        return html_results


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=8080)

#!/bin/bash

mlflow server --backend-store-uri mlruns/ --default-artifact-root mlruns/ --host 0.0.0.0 --port 5000 & python ml_flow_script.py && fg

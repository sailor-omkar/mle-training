import argparse
import os
from pathlib import Path

import pandas as pd

FILE_PATH = Path(__file__)
PROJECT_DIR = FILE_PATH.resolve().parents[2]
ARTIFACTS_DIR = os.path.join(PROJECT_DIR, "artifacts")
DATA_DIR = os.path.join(PROJECT_DIR, "datasets")

import house_price_prediction.data_scripts.create_features as cf
from house_price_prediction.utility_scripts.log_config import generate_logger
from house_price_prediction.utility_scripts.utils import Utils
from sklearn.metrics import mean_squared_error


class ScoreModels:
    def __init__(self) -> None:
        self.utility_obj = Utils()
        pass

    def model_predict(self, queries, model_path, model_name):
        """model for prediction.

        Args:
            features (_type_): queries on which label has to predited
            model_path (_type_): path to model data

        Returns:
            int: prediction
        """
        model = self.utility_obj.load_pickle(model_path, model_name)
        predictions = model.predict(queries)
        return predictions

    def root_mean_square_error(self, predictions, actual_values):
        """find out root mean square error.

        Args:
            predictions (_type_): _description_
            actual_values (_type_): _description_

        Returns:
            int: root mean square error.
        """

        rmse = mean_squared_error(actual_values, predictions)
        return rmse


if __name__ == "__main__":

    print("Enter model of choice")
    logger = generate_logger("score_scripts", "score.log")
    parser = argparse.ArgumentParser(description="To score models on data.")
    parser.add_argument(
        "-d",
        "--data_path",
        help="Provide valid data path.",
        default=os.path.join(DATA_DIR, "processed"),
    )

    parser.add_argument(
        "-p", "--model_path", help="Provide model path.", default=ARTIFACTS_DIR
    )

    parser.add_argument(
        "-a", "--model_name", help="Provide model name.", default="best_model.pickle"
    )

    parser.add_argument(
        "-f", "--file_name", help="Provide a test file name.", default="test.csv"
    )

    model = ScoreModels()
    args = parser.parse_args()
    utility = Utils()
    test_data = utility.load_project_data(
        os.path.join(DATA_DIR, "processed"), args.file_name
    )

    X_test = test_data.drop("median_house_value", axis=1)
    y_test = test_data["median_house_value"].copy()
    X_test_num = X_test.drop("ocean_proximity", axis=1)

    imputer_obj = cf.Imputer()

    imputer_name = "numeric_imputer.pickle"
    artifact_path = os.path.join(ARTIFACTS_DIR)
    imputer_model = utility.load_pickle(artifact_path, imputer_name)
    imputed_data = imputer_obj.impute_numeric_values(X_test_num, imputer_model, logger)

    data_tr = pd.DataFrame(
        imputed_data, columns=X_test_num.columns, index=test_data.index
    )

    vals = [
        ["rooms_per_household", "total_rooms", "households"],
        ["bedrooms_per_room", "total_bedrooms", "total_rooms"],
        ["population_per_household", "population", "households"],
    ]

    feature_engineer = cf.FetureEngineer()

    for val in vals:
        data_tr = feature_engineer.calculate_ratios(
            data=data_tr,
            new_var=val[0],
            numerator=val[1],
            denominator=val[2],
            logger=logger,
        )

    data_cat = test_data[["ocean_proximity"]]
    X_test_processed = data_tr.join(pd.get_dummies(data_cat, drop_first=True))
    final_predictions = model.model_predict(
        X_test_processed, args.model_path, args.model_name
    )

    rmse = model.root_mean_square_error(final_predictions, y_test)
    print(rmse)

# this file contains sript to create features
import argparse
import os
import pickle
import sys
from pathlib import Path
from pprint import pprint

import house_price_prediction_omkar.utility_scripts.utils as ut
import mlflow
import numpy as np
import pandas as pd
from house_price_prediction_omkar.utility_scripts.log_config import generate_logger
from pip import main
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FILE_PATH = Path(__file__)
PROJECT_DIR = FILE_PATH.resolve().parents[2]
ARTIFACTS_DIR = os.path.join(PROJECT_DIR, "artifacts")
DATA_DIR = os.path.join(PROJECT_DIR, "datasets")


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):
        rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[
                X, rooms_per_household, population_per_household, bedrooms_per_room
            ]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


class ProcessPipeline:
    def __init__(self) -> None:
        pass

    def process_data(self, data, label, cat_attribs, num_attribs, new_features):

        data_x = data.drop([label], axis=1)
        with mlflow.start_run(run_name='CREATE_FEATURE', nested=True) as create_feature_run:
            mlflow.log_param("child", "yes")
            imputer = SimpleImputer(strategy="median")
            attr_adder = CombinedAttributesAdder()
            std_scaler = StandardScaler()
            num_pipeline = Pipeline(
                [
                    ("imputer", imputer),
                    ("attribs_adder", attr_adder),
                    ("std_scaler", std_scaler),
                ]
            )

            onehot = OneHotEncoder(handle_unknown="ignore")
            col_transformers = ColumnTransformer(
                [
                    ("num", num_pipeline, num_attribs),
                    ("cat", onehot, [cat_attribs])]
            )

            clf = Pipeline(steps=[("preprocessor", col_transformers)])
            train_x_prepared = clf.fit_transform(data_x)
            categorical_columns = (
                clf.named_steps["preprocessor"]
                .transformers_[1][1]
                .get_feature_names([cat_attribs])
            )

            num_attribs.extend(new_features)

            num_attribs.extend(categorical_columns)

            train_prepared = pd.DataFrame(
                                            train_x_prepared,
                                            columns=num_attribs)
            train_prepared[label] = data[label]

            mlflow.log_param(key="imputing_strategy", value="median")
            mlflow.sklearn.log_model(imputer, "imputer")
            mlflow.sklearn.log_model(onehot, "onehot_encoder")
        return train_prepared


class Imputer:
    def __init__(self) -> None:
        """function to initialize the object."""
        pass

    def fit_imputer(self, data, imputer_path, imputer_name, logger):
        """
        function to fit imputer

        Args:
            data (_type_): pandas dataframe.
            imputer_path (str): path to folder imputer.
            imputer_name (str): file name of imputer.
            logger (_type_): logging object.
        """
        logger.debug("fitting the imputer over the data.")
        imputer = SimpleImputer(strategy="median")
        imputer.fit(data)

        if not os.path.exists(imputer_path):
            os.makedirs(imputer_path)

        pickle.dump(imputer, open(os.path.join(imputer_path, imputer_name), "wb"))
        logger.debug("successfully created an imputer.")

    def impute_numeric_values(self, data, imputer_obj, logger):
        """
        function to impute numeric values.

        Args:
            data (_type_): pandas dataframe.
            imputer_obj (_type_): imputer object.
            logger (_type_): logging object.
        Returns:
            _type_: numpy array of imputed data.
        """

        logger.debug("imputing the numerical data.")
        if isinstance(imputer_obj, SimpleImputer):
            transformed_data = imputer_obj.transform(data)
            return transformed_data


class FetureEngineer:
    def __init__(self) -> None:
        """
        function to initialize object.
        """
        pass

    def create_binning_feature(
        self, data, ref_feature, new_feature, bins, labels, logger
    ):
        """
        function to add new labels by using
        binning technique on basis of a feature.

        Args:
            data (_type_): pandas dataframe.
            ref_feature (str): name of reference variable.
            new_feature (str): name of new variable.
            bins (list): list of bins.
            labels (list): label of bins.
            logger (_type_): logging object.

        Returns:
            _type_: pandas dataframe with new feature.
        """
        try:
            data[new_feature] = pd.cut(data[ref_feature], bins=bins, labels=labels)
        except Exception:
            logger.error("Error while creating bins.")

        logger.debug("Successfully created binning feature.")
        return data

    def get_proportions(self, data, label, logger):
        """
        function which returns proportion of each label
        category.

        Args:
            data (_type_): pandas dataframe.
            label (str): data labels.
            logger (_type_): logging object.

        Returns:
            _type_: proportion of a label in whole data.
        """
        logger.debug("calculating the proportion.")
        return data[label].value_counts() / len(data)

    def calculate_ratios(self, data, new_var, numerator, denominator, logger):
        """
        function which calculates ratios between features

        Args:
            data (_type_): pandas dataframe.
            new_var (str): new variable name.
            numerator (str): numerator variable name.
            denominator (str): denominator variable name.
            logger (_type_): logging object.

        Returns:
            _type_: pandas dataframe.
        """

        logger.debug("calculating the ratio.")
        data[new_var] = data[numerator] / data[denominator]

        return data


def main():

    logger = generate_logger("data_scripts", "create_feature.log")
    parser = argparse.ArgumentParser(description="To create new features")

    parser.add_argument(
        "-d",
        "--data_path",
        help="Provide valid data path.",
        default=os.path.join(DATA_DIR, "processed"),
    )

    parser.add_argument(
        "-f", "--file_name", help=" Provide a file name.", default="train.csv"
    )

    args = parser.parse_args()

    feature_engineer = FetureEngineer()
    utility = ut.Utils()
    imputer = Imputer()

    data = utility.load_project_data(
                                        data_path=args.data_path,
                                        file_name=args.file_name)

    num_attribs = list(data.columns)
    num_attribs.remove("ocean_proximity")
    num_attribs.remove("median_house_value")
    cat_attribs = "ocean_proximity"

    new_features = [
        "rooms_per_household",
        "bedrooms_per_room",
        "population_per_household",
    ]

    proc_pipe = ProcessPipeline()
    train_prepared = proc_pipe.process_data(
        data=data,
        label="median_house_value",
        cat_attribs=cat_attribs,
        num_attribs=num_attribs,
        new_features=new_features,
    )

    utility.store_dataframe(train_prepared, args.data_path, "train.csv")

if __name__ == "__main__":
    main()

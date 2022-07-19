# this file contains sript to create features
import argparse
import os
import pickle
import sys
from pathlib import Path
from pprint import pprint

import pandas as pd
import house_price_prediction.utility_scripts.utils as ut
from house_price_prediction.utility_scripts.log_config import generate_logger
from pip import main
from sklearn.impute import SimpleImputer

FILE_PATH = Path(__file__)
PROJECT_DIR = FILE_PATH.resolve().parents[2]
ARTIFACTS_DIR = os.path.join(PROJECT_DIR, "artifacts")
DATA_DIR = os.path.join(PROJECT_DIR, "datasets")


class Imputer:
    def __init__(self) -> None:
        """function to initialize the object.
        """
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
        logger.debug('fitting the imputer over the data.')
        imputer = SimpleImputer(strategy="median")
        imputer.fit(data)

        if not os.path.exists(imputer_path):
            os.makedirs(imputer_path)

        pickle.dump(imputer, open(
                                    os.path.join(imputer_path, imputer_name),
                                    "wb"))
        logger.debug('successfully created an imputer.')

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

        logger.debug('imputing the numerical data.')
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
                                self,
                                data,
                                ref_feature,
                                new_feature,
                                bins,
                                labels,
                                logger):
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
            data[new_feature] = pd.cut(
                                        data[ref_feature],
                                        bins=bins,
                                        labels=labels)
        except Exception:
            logger.error('Error while creating bins.')

        logger.debug('Successfully created binning feature.')
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
        logger.debug('calculating the proportion.')
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

        logger.debug('calculating the ratio.')
        data[new_var] = data[numerator] / data[denominator]

        return data


if __name__ == "__main__":

    logger = generate_logger("data_scripts", "create_feature.log")
    parser = argparse.ArgumentParser(description="To create new features")

    parser.add_argument(
        "-d",
        "--data_path",
        help="Provide valid data path.",
        default=os.path.join(DATA_DIR, "processed"),
    )

    parser.add_argument(
        "-f",
        "--file_name",
        help=" Provide a file name.",
        default="train.csv"
    )

    args = parser.parse_args()

    feature_engineer = FetureEngineer()
    utility = ut.Utils()
    imputer = Imputer()

    data = utility.load_project_data(
                                        data_path=args.data_path,
                                        file_name=args.file_name)
    numeric_data = data.drop(["ocean_proximity", "median_house_value"], axis=1)

    imputer = Imputer()
    imputer_name = "numeric_imputer.pickle"
    artifact_path = os.path.join(ARTIFACTS_DIR)
    imputer.fit_imputer(numeric_data, artifact_path, imputer_name, logger)
    imputer_obj = utility.load_pickle(artifact_path, imputer_name)
    imputed_data = imputer.impute_numeric_values(
                                                numeric_data,
                                                imputer_obj,
                                                logger)

    data_tr = pd.DataFrame(
                            imputed_data,
                            columns=numeric_data.columns,
                            index=data.index)

    data_tr['median_house_value'] = data['median_house_value']

    vals = [
        ["rooms_per_household", "total_rooms", "households"],
        ["bedrooms_per_room", "total_bedrooms", "total_rooms"],
        ["population_per_household", "population", "households"],
    ]

    for val in vals:
        data_tr = feature_engineer.calculate_ratios(
            data=data_tr, new_var=val[0], numerator=val[1], denominator=val[2],
            logger=logger
        )

    data_cat = data[["ocean_proximity"]]
    data_prepared = data_tr.join(pd.get_dummies(data_cat, drop_first=True))

    utility.store_dataframe(data_prepared, args.data_path, "train.csv")

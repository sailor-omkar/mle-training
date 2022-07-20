# this file compares various splitting methods
# and selects the approproiate one among random and stratified split.
import argparse
import os
from cgi import test
from pathlib import Path
from pprint import pprint

import house_price_prediction.data_scripts.create_features as cf
import house_price_prediction.utility_scripts.utils as ut
import mlflow
import numpy as np
import pandas as pd
from house_price_prediction.utility_scripts.log_config import generate_logger
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

FILE_PATH = Path(__file__)
PROJECT_DIR = FILE_PATH.resolve().parents[2]
ARTIFACTS_DIR = os.path.join(PROJECT_DIR, "artifacts")
DATA_DIR = os.path.join(PROJECT_DIR, "datasets")


class SplitData:
    __slots__ = [
        "data",
        "rand_train_set",
        "rand_test_set",
        "strat_train_set",
        "strat_test_set",
        "feature_eng_obj",
        "compare_props",
        "split_least_error",
    ]

    def __init__(self) -> None:

        """
        function to initialize object.
        """
        self.feature_eng_obj = cf.FetureEngineer()
        pass

    def load_project_data(self, data_path, file_name, logger):
        """
        function to load housing data in pandas dataframe.

        Args:
            data_path (str): path of data folder.
            file_name (str): data file name.
            logger (_type_): logger to log into file.
        """

        logger.debug('loading project data.')
        csv_path = os.path.join(data_path, file_name)
        try:
            self.data = pd.read_csv(csv_path)
        except Exception:
            logger.error('Exception while loading data')
            raise
        return True

    def normal_split(self, test_size, logger):
        """function to perform normal train test split.

        Args:
            logger (_type_): logger object.
            test_size (float): test set size.

        Returns:
            list: List containing train-test split of inputs.
        """

        logger.debug('performing normal train test split')

        self.rand_train_set, self.rand_test_set = train_test_split(
            self.data, test_size=test_size, random_state=42
        )

        return self.rand_train_set, self.rand_test_set

    def startified_split(
                        self,
                        label,
                        n_splits,
                        test_size,
                        random_state,
                        logger):
        """

        Args:
            label (_type_): reference data label.
            logger (_type_): logger object.
            n_splits (int): number of splits.
            test_size (float): test split size.
            random_state (int): random number.
        """

        logger.debug('performing stratified train test split')

        with mlflow.start_run(run_name='DATA_SPLIT', nested=True) as data_split_run:
            mlflow.log_param("child", "yes")
            split = StratifiedShuffleSplit(
                n_splits=n_splits,
                test_size=test_size,
                random_state=random_state)

            for train_index, test_index in split.split(
                                                        self.data,
                                                        self.data[label]):

                self.strat_train_set = self.data.loc[train_index]
                self.strat_test_set = self.data.loc[test_index]

            mlflow.log_param(key="n_splits", value=n_splits)
            mlflow.log_param(key="test_size", value=test_size)
            mlflow.log_param(key="split_random_state", value=random_state)
            mlflow.sklearn.log_model(split, "stratified_splitter")

    def compare_splits(self, label, logger):
        """
        comparing distribution of a label
        in different type of splits.

        Args:
            label (str): name of the column.

        Returns:
            str: splitting method which more
                adheres to the actual distribution.
        """

        self.compare_props = pd.DataFrame(
            {
                "Overall": self.feature_eng_obj.get_proportions(
                    data=self.data, label=label, logger=logger
                ),
                "Stratified": self.feature_eng_obj.get_proportions(
                    data=self.strat_test_set, label=label, logger=logger
                ),
                "Random": self.feature_eng_obj.get_proportions(
                    data=self.rand_test_set, label=label, logger=logger
                )
            }
        ).sort_index()

        self.compare_props["Rand. %error"] = (
                                                (
                                                    100 * (
                                                        self.compare_props["Random"] /
                                                        self.compare_props["Overall"])
                                                ) - 100)

        self.compare_props["Strat. %error"] = (
                                                (
                                                    100 * (
                                                        self.compare_props["Stratified"] /
                                                        self.compare_props["Overall"])
                                                ) - 100)

        mean_error_strat_split = abs(self.compare_props["Strat. %error"].mean())

        mean_error_random_split = abs(self.compare_props["Rand. %error"].mean())

        self.split_least_error = (
            "stratified"
            if (mean_error_strat_split <= mean_error_random_split)
            else "random"
        )
        logger.info(self.split_least_error + " is the method with least"
                    " splitting error.")

        return self.split_least_error

    def save_split_data(self, data_path, logger):
        """function to split data.

        Args:
            data_path (str): path to store the splitted files.
            logger (_type_): logging cobject.

        Returns:
            bool: True, if file is saved.
        """

        train_file_path = os.path.join(data_path, "train.csv")
        test_file_path = os.path.join(data_path, "test.csv")

        try:
            if self.split_least_error == "stratified":
                self.strat_test_set.to_csv(test_file_path, index=False)
                self.strat_train_set.to_csv(train_file_path, index=False)
            else:
                self.rand_test_set.to_csv(test_file_path, index=False)
                self.rand_train_set.to_csv(train_file_path, index=False)
        except Exception:
            logger.error('Error while saving split data in csv')
            raise

        return True


def main():

    logger = generate_logger("data_scripts", "split.log")

    parser = argparse.ArgumentParser(
        description="To compare data splitting strategies"
    )

    parser.add_argument(
        "-d",
        "--data_path",
        help="Provide valid data path.",
        default=os.path.join(DATA_DIR, "raw"),
    )

    parser.add_argument(
        "-f", "--file_name",
        help="Provide a file name.",
        default="housing.csv"
    )

    parser.add_argument(
        "-o",
        "--output_folder",
        help=" Provide output data path.",
        default=os.path.join(DATA_DIR, "processed"),
    )

    args = parser.parse_args()

    logger.debug('Creating split data object.')
    data_splitter = SplitData()
    data_splitter.load_project_data(
                                    data_path=args.data_path,
                                    file_name=args.file_name,
                                    logger=logger)

    # creating income category variable for stratified splitting
    data_splitter.data = data_splitter.feature_eng_obj.create_binning_feature(
        data_splitter.data,
        ref_feature="median_income",
        new_feature="income_cat",
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
        logger=logger
    )

    # creating normal train test split
    data_splitter.normal_split(test_size=0.2, logger=logger)

    # performing stratified split on basis of income category
    data_splitter.startified_split(
        label="income_cat",
        n_splits=1,
        test_size=0.2,
        random_state=4,
        logger=logger
    )

    best_split_method = data_splitter.compare_splits(
                                                        label="income_cat",
                                                        logger=logger)
    print("Best split is obtained by :", best_split_method)

    # removing income category from data
    for set_ in (data_splitter.strat_train_set, data_splitter.strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    data_splitter.save_split_data(
                                    data_path=args.output_folder,
                                    logger=logger)

if __name__ == "__main__":
    main()

# in this script we will train various models and
# select best one for prediction
import argparse
import os
from pathlib import Path
from pprint import pprint

import numpy as np
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor

from house_price_prediction.utility_scripts.utils import Utils
from house_price_prediction.utility_scripts.log_config import generate_logger

FILE_PATH = Path(__file__)
PROJECT_DIR = FILE_PATH.resolve().parents[2]
ARTIFACTS_DIR = os.path.join(PROJECT_DIR, "artifacts")
DATA_DIR = os.path.join(PROJECT_DIR, "datasets")

class TrainData:
    def __init__(self, features, labels, output_folder) -> None:
        self.features = features
        self.labels = labels
        self.output_folder = output_folder
        self.utility = Utils()
        pass

    def linear_regression(self, logger):
        """function to fit linear regressor on the model.

        Args:
            logger (_type_): logging object.

        Returns:
            str: returns model file name and rmse score.
        """
        lin_reg = LinearRegression()

        # fit linear regressor on data
        logger.debug('training linear regression model.')
        lin_reg.fit(self.features, self.labels)

        # make predictions from the model
        logger.debug('making predictions.')
        predictions = lin_reg.predict(self.features)

        # get rmse
        logger.debug('mean square error.')
        lin_mse = mean_squared_error(self.labels, predictions)
        lin_rmse = np.sqrt(lin_mse)

        logger.debug('storing model in a pickle file.')
        self.utility.store_pickle(
                                    lin_reg,
                                    self.output_folder,
                                    file_name="linear_regressor.pickle")
        return "linear_regressor.pickle", lin_rmse

    def decision_tree(self, logger):
        """function to fit decision tree model on data.

        Args:
            logger (_type_): logging object.

        Returns:
            str: returns model file name and rmse score.
        """
        # training decision tree regression model

        logger.debug('training decision tree regression model.')
        tree_reg = DecisionTreeRegressor(random_state=42)
        tree_reg.fit(self.features, self.labels)

        logger.debug('predicting from decision tree regression model.')
        predictions = tree_reg.predict(self.features)
        tree_mse = mean_squared_error(self.labels, predictions)
        tree_rmse = np.sqrt(tree_mse)

        self.utility.store_pickle(
                                    tree_reg,
                                    self.output_folder,
                                    file_name="decision_tree.pickle")
        return "decision_tree.pickle", tree_rmse

    def random_forest_random_search(self, logger):
        """function to train random forest regressor
            using random search cv on the data.

        Args:
            logger (_type_): logging object.

        Returns:
            str: returns model file name and rmse score.
        """

        param_distribs = {
            "n_estimators": randint(low=1, high=200),
            "max_features": randint(low=1, high=8),
        }

        logger.debug('initialising random forest model.')
        forest_reg = RandomForestRegressor(random_state=42)

        logger.debug('initialising random search cv.')
        rnd_search = RandomizedSearchCV(
            forest_reg,
            param_distributions=param_distribs,
            n_iter=10,
            cv=5,
            scoring="neg_mean_squared_error",
            random_state=42,
        )

        logger.debug('fitting random forest tree on the training data.')
        rnd_search.fit(self.features, self.labels)

        logger.debug('predicting from model.')
        prediction = rnd_search.predict(self.features)
        rn_search_mse = mean_squared_error(self.labels, prediction)
        rn_search_mse = np.sqrt(rn_search_mse)

        logger.debug('storing the random forest model in a pickle.')
        self.utility.store_pickle(
                                rnd_search,
                                self.output_folder,
                                file_name="random_forest_random_search.pickle")
        return "random_forest_random_search.pickle", rn_search_mse

    def random_forest_grid_search(self, logger):
        """function to train random forest regressor
            using grid search cv on the data.


        Args:
            logger (_type_): logging object.

        Returns:
            str: returns model file name and rmse score.
        """

        param_grid = [
                            # try 12 (3×4) combinations of hyperparameters
                            {
                                'n_estimators': [3, 10, 30],
                                'max_features': [2, 4, 6, 8]},
                            # then try 6 (2×3) combinations with
                            # bootstrap set as False
                            {
                                'bootstrap': [False],
                                'n_estimators': [3, 10],
                                'max_features': [2, 3, 4]},
                    ]

        logger.debug('initialising random forest regressor model.')
        forest_reg = RandomForestRegressor(random_state=42)

        # train across 5 folds, that's a total of
        # (12+6)*5=90 rounds of training
        logger.debug('initialising grid search cv.')
        grid_search = GridSearchCV(
                                    forest_reg, param_grid, cv=5,
                                    scoring='neg_mean_squared_error',
                                    return_train_score=True)

        logger.debug('fitting grid search cv.')
        grid_search.fit(self.features, self.labels)

        logger.debug('predicting from model.')
        prediction = grid_search.predict(self.features)
        grid_search_mse = mean_squared_error(self.labels, prediction)
        grid_search_mse = np.sqrt(grid_search_mse)

        logger.debug('storing model as a pickle.')
        self.utility.store_pickle(
                                grid_search,
                                self.output_folder,
                                file_name="random_forest_grid_search.pickle")

        return "random_forest_grid_search.pickle", grid_search_mse


if __name__ == "__main__":

    logger = generate_logger("train_scripts", "train.log")
    parser = argparse.ArgumentParser(description="To train models on data")
    parser.add_argument(
                        "-d",
                        "--data_path",
                        help="Provide valid data path.",
                        default=os.path.join(DATA_DIR, "processed"))

    parser.add_argument(
                        "-f",
                        "--file_name",
                        help="Provide a file name.",
                        default="train.csv")

    parser.add_argument(
                        "-o",
                        "--output_folder_path",
                        help="Provide a output folder name.",
                        default=ARTIFACTS_DIR)

    parser.add_argument(
                        "-l",
                        "--label_column",
                        help="Provide label column name.",
                        default="median_house_value")

    args = parser.parse_args()

    utility = Utils()
    data = utility.load_project_data(args.data_path, args.file_name)

    features = data.drop(args.label_column, axis=1)
    label = data[args.label_column].copy()

    logger.debug('initialising trainer object.')
    trainer = TrainData(
                        features=features, labels=label,
                        output_folder=args.output_folder_path)

    model_scores = []
    print('training linear regression...')
    model_scores.append(trainer.linear_regression(logger=logger))
    print('training decision tree...')
    model_scores.append(trainer.decision_tree(logger=logger))
    print('training random forest random cv...')
    model_scores.append(trainer.random_forest_random_search(logger=logger))
    print('training random forest grid cv...')
    model_scores.append(trainer.random_forest_grid_search(logger=logger))

    logger.debug('saving best model.')
    best_model_name = min(model_scores, key=lambda t: t[1])[0]
    os.rename(
                os.path.join(args.output_folder_path, best_model_name),
                os.path.join(args.output_folder_path, 'best_model.pickle'))

# this script performs exploratory data analysis.
import argparse
import os
from pathlib import Path

from house_price_prediction.utility_scripts.log_config import generate_logger
import house_price_prediction.data_scripts.create_features as cf
import pandas as pd

FILE_PATH = Path(__file__)
PROJECT_DIR = FILE_PATH.resolve().parents[2]
ARTIFACTS_DIR = os.path.join(PROJECT_DIR, "artifacts")
DATA_DIR = os.path.join(PROJECT_DIR, "datasets")


class ExploreData:
    def __init__(self) -> None:
        """
        function to initialize object.
        """
        self.feature_eng_obj = cf.FetureEngineer()
        pass

    def load_project_data(self, data_path, file_name, logger):
        """function to load data in pandas dataframe.

        Args:
            data_path (str): path to data folder.
            file_name (str): name of data file
        """

        logger.debug('loading project data.')
        csv_path = os.path.join(data_path, file_name)
        try:
            self.data = pd.read_csv(csv_path)
        except Exception:
            logger.error('Exception while loading data')
            raise
        return True

    def scatter_plot(self, x, y, logger, **kwargs):
        """function returns scatter plot between the features.

        Args:
            x (str): x label feature
            y (str): y label feature

        Returns:
            _type_: scatter plot between the provided x, y labels
        """
        logger.debug('creating scatter plot matrix')
        return self.data.plot(kind="scatter", x=x, y=y, **kwargs)

    def correlation_matrix(self, logger, **kwargs):
        """function returns correlation between features.

        Returns:
            _type_: correlation matrix
        """
        logger.debug('creating correlation matrix')
        return self.data.corr()


if __name__ == "__main__":

    logger = generate_logger("data_scripts", "explore.log")
    parser = argparse.ArgumentParser(description="To explore data")
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

    data_explorer = ExploreData()
    status = data_explorer.load_project_data(
                    data_path=args.data_path,
                    file_name=args.file_name,
                    logger=logger)

    # see scatter plot between latitude and longitude
    data_explorer.scatter_plot(
                                x="longitude",
                                y="latitude",
                                alpha=0.1,
                                logger=logger)

    # see covariance matrix
    corr_matrix = data_explorer.correlation_matrix(logger=logger)
    corr_matrix["median_house_value"].sort_values(ascending=False)

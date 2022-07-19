# this file contains script to download data
import argparse
import os
import sys
import tarfile
from pathlib import Path

from house_price_prediction.utility_scripts.log_config import generate_logger
from six.moves import urllib

FILE_PATH = Path(__file__)
PROJECT_DIR = FILE_PATH.resolve().parents[2]
ARTIFACTS_DIR = os.path.join(PROJECT_DIR, "artifacts")
DATA_DIR = os.path.join(PROJECT_DIR, "datasets")


class Data:

    __slots__ = ("data_url", "data_folder_name")

    def __init__(self, data_url) -> None:
        """
        this function returns object of this class.

        Args:
            data_url (str): a valid url from which data has to be downloaded.
        """
        self.data_url = data_url
        self.data_folder_name = os.path.join(DATA_DIR, "raw")
        pass

    def fetch_data(self, logger):
        """
        function to extract housing data from zip file
        and consequently store data in a folder.
        Args:
            logger (_type_): logging object.
        """
        logger.debug("Entered the fetch data function")
        os.makedirs(self.data_folder_name, exist_ok=True)
        tgz_path = os.path.join(self.data_folder_name, "data.tgz")
        try:
            urllib.request.urlretrieve(self.data_url, tgz_path)
            data_tgz = tarfile.open(tgz_path)
            data_tgz.extractall(path=self.data_folder_name)
            data_tgz.close()
        except Exception:
            logger.error("Error while extracting data")
            raise

        logger.debug("Successfully downloaded raw data")
        return True


def main():

    logger = generate_logger("data_scripts", "download.log")
    parser = argparse.ArgumentParser(
        description="To load data related to project from specified url."
    )
    parser.add_argument(
        "-u",
        "--url",
        help="Provide a valid url from which to extract data.",
        default="https://raw.githubusercontent.com/ageron/handson-ml/"
        "master/datasets/housing/housing.tgz",
    )

    args = parser.parse_args()
    data_loader = Data(args.url)
    response = data_loader.fetch_data(logger=logger)


if __name__ == "__main__":
    main()

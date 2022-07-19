import unittest
from ast import Assert

from house_price_prediction.data_scripts import download_data as dd
from house_price_prediction.utility_scripts.log_config import generate_logger

logger = generate_logger("data_scripts", "download.log")

class DownloadDataTest(unittest.TestCase):
    # methods in this class should start with the word test

    def test_fetch_data_invalid_url(self):
        d_obj = dd.Data('jaba')
        with self.assertRaises(ValueError):
            d_obj.fetch_data(logger=logger)

    def test_fetch_data_valid_url(self):
        d_obj = dd.Data(
                        "https://raw.githubusercontent.com/ageron/handson-ml/"
                        "master/datasets/housing/housing.tgz")

        response = d_obj.fetch_data(logger=logger)
        self.assertEqual(response, True)

# this script provides utility functions
from operator import index
import os
import pickle

import pandas as pd


class Utils:
    def __init__(self) -> None:

        """
        function to initialize object.
        """
        pass

    def load_project_data(self, data_path, file_name):
        """function to load data in pandas dataframe.

        Args:
            data_path (str): path to data folder.
            file_name (str): file name.

        Returns:
            _type_: _description_
        """

        csv_path = os.path.join(data_path, file_name)
        data = pd.read_csv(csv_path, index_col=False)
        return data

    def store_dataframe(self, data, data_path, file_name):
        """function to save pandas dataframe in csv.

        Args:
            data (_type_): _description_
            data_path (str): path to data folder.
            file_name (str): file name.
        """

        csv_path = os.path.join(data_path, file_name)
        data.to_csv(csv_path, index=False)

    def load_pickle(self, artifact_path, file_name):
        """function to load pickle file from artifact directory.

        Args:
            artifact_path (str): path to artifacts folder.
            file_name (str): model file name.

        Returns:
            _type_: _description_
        """

        pickle_path = os.path.join(artifact_path, file_name)
        print(pickle_path)
        artifact = pickle.load(open(pickle_path, "rb"))
        return artifact

    def store_pickle(self, data, artifact_path, file_name):
        """function to load pickle file from artifact directory.

        Args:
            data (_type_): model to be pickled.
            artifact_path (str): path to artifacts folder.
            file_name (str): model file name.
        """

        pickle_path = os.path.join(artifact_path, file_name)
        pickle.dump(data, open(pickle_path, "wb"))

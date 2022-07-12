# importing libraries
import numpy as np
import pandas as pd
import os
import tarfile
from six.moves import urllib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# declaring global variables (file paths)
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


class Utils():
    def __init__(self):
        return

    def fetch_housing_data(
                            self,
                            housing_url=HOUSING_URL,
                            housing_path=HOUSING_PATH):

        """
        function to extract housing data from zip file
        and consequently
        store data in a folder.
        """

        os.makedirs(housing_path, exist_ok=True)
        tgz_path = os.path.join(housing_path, "housing.tgz")
        urllib.request.urlretrieve(housing_url, tgz_path)
        housing_tgz = tarfile.open(tgz_path)
        housing_tgz.extractall(path=housing_path)
        housing_tgz.close()

    def load_housing_data(self, housing_path=HOUSING_PATH):

        """
        function to load housing data in pandas dataframe.
        """

        csv_path = os.path.join(housing_path, "housing.csv")
        return pd.read_csv(csv_path)

    def income_cat_proportions(self, data):

        """
        function which returns proportion of each income
        category.
        """
        return data["income_cat"].value_counts() / len(data)


class Feature_engineer():

    def __init__(self) -> None:
        pass

    def compte_ratios_per_houseold(self, df):

        df["rooms_per_household"] = (
                                    df["total_rooms"]
                                    / df["households"]
                                    )

        df["bedrooms_per_room"] = (
                            df["total_bedrooms"]
                            / df["total_rooms"])

        df["population_per_household"] = (
                                            df["population"]
                                            / df["households"])

        return df


# initializing objects
feature_eng = Feature_engineer()
utils = Utils()

# fetching data
utils.fetch_housing_data()
housing = utils.load_housing_data()

# creating income category variable for stratified splitting
housing["income_cat"] = pd.cut(
                                housing["median_income"],
                                bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                labels=[1, 2, 3, 4, 5])

# performing normal train test split
train_set, test_set = train_test_split(
                                        housing,
                                        test_size=0.2,
                                        random_state=42)

# performing stratified split on basis of income category
split = StratifiedShuffleSplit(
                                n_splits=1,
                                test_size=0.2,
                                random_state=42)
for train_index, test_index in split.split(
                                            housing,
                                            housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# comparing distribution of each income category in different type of splits.
compare_props = pd.DataFrame({
    "Overall": utils.income_cat_proportions(housing),
    "Stratified": utils.income_cat_proportions(strat_test_set),
    "Random": utils.income_cat_proportions(test_set),
}).sort_index()

compare_props["Rand. %error"] = (
                                100 * compare_props["Random"]
                                / compare_props["Overall"] - 100
                                )

compare_props["Strat. %error"] = (
                                    100 * compare_props["Stratified"]
                                    / compare_props["Overall"] - 100
                                )

# removing income category from data
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

# plotting training data by using longitude and latitude
housing = strat_train_set.copy()
housing.plot(kind="scatter", x="longitude", y="latitude")
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

# finding correlation of "median_house_value" with other attributes
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

# splitting features and labels
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# dropping 'ocean_proximity' and just getting numeric
#                           features for imputation purpose
imputer = SimpleImputer(strategy="median")
housing_num = housing.drop('ocean_proximity', axis=1)
imputer.fit(housing_num)
X = imputer.transform(housing_num)

# calculating numeric features again.
housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing.index)
housing_tr = feature_eng.compte_ratios_per_houseold(housing_tr)

# encoding categorical variables
housing_cat = housing[['ocean_proximity']]
housing_prepared = housing_tr.join(
                    pd.get_dummies(housing_cat, drop_first=True))

# training linear regression model on prepared housing data
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print(
    "RMSE on training data for Linear Reg. :", lin_rmse)

lin_mae = mean_absolute_error(housing_labels, housing_predictions)
print(
    "Mean absolute error on training data for Linear Reg. :", lin_mae)

# training decision tree regression model
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
print(
    "RMSE on training data for Decision tree :", tree_rmse)

# training random forest regressor by performing random search cv.
param_distribs = {
        'n_estimators': randint(low=1, high=200),
        'max_features': randint(low=1, high=8),
    }

forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(
                                forest_reg,
                                param_distributions=param_distribs,
                                n_iter=10,
                                cv=5,
                                scoring='neg_mean_squared_error',
                                random_state=42)

rnd_search.fit(housing_prepared, housing_labels)

cvres = rnd_search.cv_results_
print(
        "Following is the test score on "
        "various random params. for Rand. Forest")

for mean_score, params in zip(
                                cvres["mean_test_score"],
                                cvres["params"]
                            ):
    print(np.sqrt(-mean_score), params)

# training random forest regressor by performing grid search cv
param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {
        'n_estimators': [3, 10, 30],
        'max_features': [2, 4, 6, 8]
    },
    # then try 6 (2×3) combinations with bootstrap set as False
    {
        'bootstrap': [False],
        'n_estimators': [3, 10],
        'max_features': [2, 3, 4]
    },
  ]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training
grid_search = GridSearchCV(
                            forest_reg,
                            param_grid,
                            cv=5,
                            scoring='neg_mean_squared_error',
                            return_train_score=True
                            )
grid_search.fit(housing_prepared, housing_labels)

cvres = grid_search.cv_results_
print(
        "Following are the test scores on "
        "grid search on various params. for Rand. Forest")
for mean_score, params in zip(
                            cvres["mean_test_score"],
                            cvres["params"]):
    print(np.sqrt(-mean_score), params)

feature_importances = grid_search.best_estimator_.feature_importances_
sorted(zip(
            feature_importances,
            housing_prepared.columns), reverse=True)

# Preprocessing the test data
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_num = X_test.drop('ocean_proximity', axis=1)
X_test_prepared = imputer.transform(X_test_num)
X_test_prepared = pd.DataFrame(
                                X_test_prepared,
                                columns=X_test_num.columns,
                                index=X_test.index)

X_test_prepared = feature_eng.compte_ratios_per_houseold(X_test_prepared)

# performing prediction on test data
X_test_cat = X_test[['ocean_proximity']]
X_test_prepared = X_test_prepared.join(
                                        pd.get_dummies(
                                                        X_test_cat,
                                                        drop_first=True)
                                        )

final_model = grid_search.best_estimator_
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print("Following is the final rmse score: ", final_rmse)
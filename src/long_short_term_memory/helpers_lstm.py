import numpy as np
import pandas as pd
import tensorflow as tf
import random
import datetime
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from src.common.helpers_common import extract_close_from_sequence

# set seed, so we can get the same results after rerunning several times
np.random.seed(314)
tf.random.set_seed(314)
random.seed(314)


def shuffle_in_unison(a, b, seed: int = 11) -> tuple:
    """
    Shuffle two arrays in the same way.
    :param a: input vector a
    :param b: input vector b
    :param seed: seed for random function
    :return: shuffled vectors
    """
    a, b = shuffle(a, b, random_state=seed)
    return a, b


def split_train_test_lstm(X, y, price_df: pd.DataFrame, feature_list: list,
                          initial_date_test: datetime.date, test_size: float, split_by_date: bool = True,
                          shuffle: bool = True) -> dict:
    """
    Split data into train and test datasets.
    :param X: sequence of features with associated history
    :param y: sequence of target values to predict
    :param price_df: dataframe of the price history
    :param feature_list: list of features to be used for prediction (dataframe column names)
    :param initial_date_test: initial date dataset
    :param test_size: test size for random shuffle
    :param split_by_date: split by date bool flag or random
    :param shuffle: flag to shuffle data
    :return: train and test data (dictionary)
    """

    num_features = len(feature_list)
    # get the test data (dataframe for plotting reasons)
    columns_test_df = ["close", "change %", "is positive change", "category % change"]
    test_df = price_df[columns_test_df].iloc[price_df.index >= initial_date_test].copy()
    number_test_days = len(test_df)
    X_test = X[-number_test_days:]
    y_test = y[-number_test_days:]

    # X_train = X[:number_test_days]
    # y_train = X[:number_test_days]

    # select random sample from test data
    # X_test_shuffled, y_test_shuffled = shuffle_in_unison(X_test, y_test, seed=45)
    # X_train.append(X_test_shuffled[:int(test_size*len(X_test_shuffled)])
    # y_train.append(y_test_shuffled[:int(test_size*len(X_test_shuffled)])
    # X_train, y_train = shuffle_in_unison(X_train, y_train, seed=45)
    if split_by_date:
        # split the dataset into training & testing sets by date (not randomly splitting)
        X_train = X[:number_test_days]
        y_train = X[:number_test_days]
        if shuffle:
            # shuffle the datasets for training (if shuffle parameter is set)
            X_train, y_train = shuffle_in_unison(X_train, y_train)
            X_test, y_test = shuffle_in_unison(X_test, y_test)
    else:
        # split the dataset randomly
        X_train, _, y_train, _ = train_test_split(X, y, test_size=test_size,
                                                            shuffle=shuffle, random_state=2)

    # extract the close from the X_test
    close_test = extract_close_from_sequence(X_test, feature_list)
    # Utest: (price_df["close"].iloc[-number_test_days:].values == close_test).all()

    # get the list of test set dates.
    # dates = X_test[:, -1, -1]
    # # retrieve test features from the original dataframe
    # result["test_df"] = df.loc[dates]
    # # remove duplicated dates in the testing dataframe
    # result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(keep='first')]
    # remove dates from the training/testing sets & convert to float32
    X_train = X_train[:, :, :num_features].astype(np.float32)
    X_test = X_test[:, :, :num_features].astype(np.float32)

    train_test_data_dict = {"X_train": X_train,
                            "y_train": y_train,
                            "X_test": X_test,
                            "y_test": y_test,
                            "test_df": test_df,
                            "close_test": close_test}

    return train_test_data_dict


def calc_bias(model, X_features: np.array, y_expected: np.array, column_scaler: dict, use_bias: bool, scale: bool) -> float:
    bias = 0
    if use_bias:
        y_predicted = model.predict(X_features)
        if scale:
            y_expected = np.squeeze(column_scaler["close"].inverse_transform(np.expand_dims(y_expected, axis=0)))
            y_predicted = np.squeeze(column_scaler["close"].inverse_transform(y_predicted))
        price_difference = [y_expected[i] - y_predicted[i] for i in range(len(y_predicted))]
        bias = np.sign(np.mean(price_difference)) * np.mean(np.abs(price_difference))
    return bias


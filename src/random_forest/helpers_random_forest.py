import copy
import datetime
import pickle
import pandas as pd
from ..common.helpers_common import (flatten,
                                     split_dataset_with_index,
                                     split_dataset_with_date,
                                     compute_precision_recall_specificity,
                                     stats_cumulative_sliding_train,
                                     check_prediction_probability_binary)
from sklearn.ensemble import RandomForestClassifier


PRINT = False


def create_and_train_model(train_data: pd.DataFrame, predictors: list,
                           parameters_model: dict) -> RandomForestClassifier:
    """
    Train a Machine Learning model (scikit-learn).
    :param train_data: full dataset of training data
    :param predictors: list of column names of the dataset used as input for the ML model
    :param parameters_model: dictionary of the model parameters
    :return: trained model
    """
    # Create the model
    model = RandomForestClassifier(n_estimators=parameters_model["n_estimators"],  # number of trees: the higher, the better the accuracy
                                   min_samples_split=parameters_model["n_samples_split"],  # the higher, the less accurate, but the less overfits
                                   random_state=1,  # if 1, same initialization
                                   min_samples_leaf=parameters_model["n_min_samples_leaf"],
                                   # criterion=,# "squared_error", "absolute_error", "friedman_mse", "poisson"
                                   n_jobs=-1)  # number of cores to be used (-1: max number of cores)

    # Train the model
    model.fit(train_data[predictors], train_data["Target"])

    return copy.deepcopy(model)


def train_and_backtest(data: pd.DataFrame, predictors: list,
                       parameters_model: dict,
                       days_initial_train: int = 2500, days_test: int = 250,
                       threshold_probability_positive: float = .6) -> tuple:
    """
    Function to backtest the model.
    We train for the first days_initial_train days, and we test the following days_test days.
    Then we train for the (days_initial_train + k*days_test) days, and we test the following days_test days.
    NOTE: this is just a backtest to test all the possible successive trainings.
    :param data: dataset
    :param predictors: list of predictors (column names as input for the model)
    :param parameters_model: dictionary of parameters of the model
    :param days_initial_train: length of the first training period
    :param days_test: length of the test period
    :param threshold_probability_positive: probability threshold to output class 1 from the model
    """
    score_cumulative_train = {"Target": [], "Prediction": []}
    number_trading_days = data.shape[0]

    # Expand the training dataset
    for i in range(days_initial_train, number_trading_days, days_test):
        # Split dataset into train and test datasets
        train_dataset, test_dataset = split_dataset_with_index(data, 0, i, days_test)

        # Train the model with the train dataset
        trained_model = create_and_train_model(train_dataset, predictors, parameters_model)

        # Test the model with the test dataset
        # a = model.predict_proba(test_data[predictors])
        # predictions = model.predict(test_data[predictors])  # predict the result (0 or 1)
        predictions = trained_model.predict_proba(test_dataset[predictors])  # predict the probability of each possible class [0, 1]
        predicted_classes = check_prediction_probability_binary(predictions[:, 1], threshold_probability_positive)
        score_cumulative_train["Target"].append(test_dataset["Target"])
        score_cumulative_train["Prediction"].append(predicted_classes)

    # Slide the training windows with a fixed size
    score_sliding_train = {"Target": [], "Prediction": []}
    k = 0
    for i in range(days_initial_train, number_trading_days, days_test):
        # Split dataset into train and test datasets
        start_train_idx = k * days_test
        train_dataset, test_dataset = split_dataset_with_index(data, start_train_idx, i, days_test)

        # Train the model with the train dataset
        trained_model = create_and_train_model(train_dataset, predictors, parameters_model)

        # Test the model with the test dataset
        # a = model.predict_proba(test_data[predictors])
        # predictions = model.predict(test_data[predictors])  # predict the result (0 or 1)
        predictions = trained_model.predict_proba(test_dataset[predictors])  # predict the probability of each possible class [0, 1]
        predicted_classes = check_prediction_probability_binary(predictions[:, 1], threshold_probability_positive)
        score_sliding_train["Target"].append(test_dataset["Target"])
        score_sliding_train["Prediction"].append(predicted_classes)
        k += 1

    return score_cumulative_train, score_sliding_train


def train_and_deploy_random_forest(data: pd.DataFrame, predictors: list,
                                   parameters_random_forest: dict,
                                   start_date_training: datetime, end_date_training: datetime,
                                   filename: str,
                                   threshold_probability_positive: float = .6):
    """
    Train final model and save model parameters.
    :param data: full dataset
    :param filename: filename for parameters
    :param predictors: list of predictors
    :param parameters_random_forest: dictionary of the parameters of the model
    :param start_date_training: initial date for training
    :param end_date_training: final date for training
    :param threshold_probability_positive: probability to accept a positive class as positive
    :return:
    """
    model = RandomForestClassifier(n_estimators=parameters_random_forest["n_estimators"],
                                   min_samples_split=parameters_random_forest["n_samples_split"],
                                   random_state=1,  # if 1, same initialization
                                   min_samples_leaf=parameters_random_forest["n_min_samples_leaf"],
                                   n_jobs=-1)  # number of cores to be used (-1: max number of cores)

    # define train and test dataset
    train_dataset, test_dataset = split_dataset_with_date(data, start_date_training, end_date_training)

    # train the model
    model.fit(train_dataset[predictors], train_dataset["Target"])

    # save model
    pickle.dump(model, open(filename, "wb"))

    # load model
    model_loaded = pickle.load(open(filename, "rb"))

    # predictions = model.predict(test_data[predictors])  # predict the result (0 or 1)
    predictions = model_loaded.predict_proba(test_dataset[predictors])  # predict the probability of each possible class [0, 1]
    predicted_classes = check_prediction_probability_binary(predictions[:, 1], threshold_probability_positive)
    # calculate scores
    # PRECISION: ability of the classifier not to label as positive a sample that is negative.
    # When the model predict a positive day, it was right precision% of times
    # SPECIFICITY: ability to predict a negative class correctly.
    # When the model predict a negative day, it was right specificity% of times
    if len(test_dataset["Target"].tolist()) != len(predicted_classes):
        print("ERROR: size error")
        exit()
    precision, recall, specificity = compute_precision_recall_specificity(test_dataset["Target"].tolist(), predicted_classes)
    print(f"Precision {precision:.2f}, recall {recall:.2f} specificity {specificity:.2f}")
    print(" ")


def create_and_backtest_random_forest(dataset: pd.DataFrame, predictors_list: list,
                                      parameters_model: dict,
                                      training_days_initial: int = 2500, test_days_step: int = 250,
                                      threshold_probability_positive: float = .6) -> None:
    """
    Create Random Forest model, train and backtest. A random forest classifier was chosen because it is resistant to overfit (due to the
    numerous random trees), runs quickly and handles non-linear relationships.
    :param dataset: full dataset
    :param predictors_list: list of columns of the dataset used for predicting the target
    :param parameters_model: dictionary of parameters of the model
    :param training_days_initial: number of trading days for the first training of the backtest
    :param test_days_step: number of trading days for testing
    :param threshold_probability_positive: probability threshold to consider a positive price prediction
    :return: None
    """

    # Train and backtest (train inside backtest)
    # NOTE: this is just a backtest to test all the possible successive trainings.
    # The deployed model will be trained using a given time range only.
    cumulative_training, sliding_training = train_and_backtest(dataset, predictors_list, parameters_model,
                                                               days_initial_train=training_days_initial,
                                                               days_test=test_days_step,
                                                               threshold_probability_positive=threshold_probability_positive)

    # calculate scores
    # PRECISION: ability of the classifier not to label as positive a sample that is negative.
    # When the model predict a positive day, it was right precision% of times
    # SPECIFICITY: ability to predict a negative class correctly.
    # When the model predict a negative day, it was right specificity% of times
    if len(flatten(cumulative_training["Target"])) != len(flatten(cumulative_training["Prediction"])):
        print("ERROR: size error")
        exit()
    stats_cumulative_sliding_train(cumulative_training, sliding_training)

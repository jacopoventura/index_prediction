import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import pickle


def predict(train_data, test_data, predictors, model, threshold_probability_positive=.6):
    """
    Train a Machine Learning model (scikit-learn).
    :param train_data: full dataset of training data
    :param test_data: full dataset of test data
    :param predictors: list of column names of the dataset used as input for the ML model
    :param model: scikit-learn ML model
    :param threshold_probability_positive: threshold of probability to accept a valid prediction of positive day
    :return: dataframe with target predictions and predictions from the trained model
    """
    # train the model
    model.fit(train_data[predictors], train_data["Target"])

    # predict the data. Target: 0 if negative day, 1 if positive day
    # predictions = model.predict(test_data[predictors])  # predict the result (0 or 1)
    predictions = model.predict_proba(test_data[predictors])[:, 1]  # predict the probability of each possible class [0, 1]
    predictions[predictions >= threshold_probability_positive] = 1
    predictions[predictions < threshold_probability_positive] = 0
    predictions = pd.Series(predictions, index=test_data.index, name="Predictions")

    # combine target of test data and predictions into a dataframe
    return pd.concat([test_data["Target"], predictions], axis=1)


def backtest(data, model, predictors, days_initial_train=2500, days_test=250, threshold_probability_positive=.6):
    """
    Function to backtest the model.
    We train for the first days_initial_train days, and we test the following days_test days.
    Then we train for the (days_initial_train + k*days_test) days, and we test the following days_test days.
    NOTE: this is just a backtest to test all the possible successive trainings.
    """
    all_predictions = []
    number_trading_days = data.shape[0]

    # Expand the training dataset
    for i in range(days_initial_train, number_trading_days, days_test):
        train_dataset = data.iloc[:i].copy()
        test_dataset = data.iloc[i:(i+days_test)].copy()
        # print("Train: ", train_dataset.index[0], " to ", train_dataset.index[-1], " days: ", train_dataset.shape[0])
        # print("Test : ", test_dataset.index[0], " to ", test_dataset.index[-1], " days: ", test_dataset.shape[0])
        predictions = predict(train_dataset, test_dataset, predictors, model, threshold_probability_positive)
        all_predictions.append(predictions)
        # print("add score")

    # Slide the training windows with a fixed size
    k = 0
    for i in range(days_initial_train, number_trading_days, days_test):
        start_train_idx = k * days_test
        train_dataset = data.iloc[start_train_idx:i].copy()
        test_dataset = data.iloc[i:(i+days_test)].copy()
        print("Train: ", train_dataset.index[0], " to ", train_dataset.index[-1], " days: ", train_dataset.shape[0])
        print("Test : ", test_dataset.index[0], " to ", test_dataset.index[-1], " days: ", test_dataset.shape[0])
        predictions = predict(train_dataset, test_dataset, predictors, model, threshold_probability_positive)
        all_predictions.append(predictions)
        k += 1
        # print("add score")

    return pd.concat(all_predictions)


def train_and_deploy(data, predictors, start_date_training, end_date_training, estimators=200, sample_split=50,
                     threshold_probability_positive=.6):

    model = RandomForestClassifier(n_estimators=estimators,  # number of trees: the higher, the better the accuracy
                                   min_samples_split=sample_split,  # the higher, the less accurate, but the less overfits
                                   random_state=1,  # if 1, same initialization
                                   n_jobs=-1)  # number of cores to be used (-1: max number of cores)

    # to do: specify initial and final dates

    train_dataset = data.loc[start_date_training:end_date_training].copy()
    test_dataset = data.loc[end_date_training:].copy()

    # train the model
    model.fit(train_dataset[predictors], train_dataset["Target"])

    # save model
    filename = "RF_test.pickle"
    pickle.dump(model, open(filename, "wb"))

    # load model
    model_loaded = pickle.load(open(filename, "rb"))

    # predict the data. Target: 0 if negative day, 1 if positive day
    # predictions = model.predict(test_data[predictors])  # predict the result (0 or 1)
    predictions = model_loaded.predict_proba(test_dataset[predictors])[:, 1]  # predict the probability of each possible class [0, 1]
    predictions[predictions >= threshold_probability_positive] = 1
    predictions[predictions < threshold_probability_positive] = 0
    predictions = pd.Series(predictions, index=test_dataset.index, name="Predictions")
    predictions = pd.concat([test_dataset["Target"], predictions], axis=1)
    ps = precision_score(predictions["Target"], predictions["Predictions"])
    print("Precision to predict a positive day (?): ", ps, "\n")


def create_and_test_random_forest(dataset, predictors_list,
                                  estimators=200, sample_split=50,
                                  training_days_initial=2500, test_days_step=250, threshold_probability_positive=.6):
    """
    Create Random Forest model, train and backtest.
    :param dataset: full dataset
    :param predictors_list: list of columns of the dataset used for predicting the target
    :param estimators: number of estimators of the random forest model
    :param sample_split: minimum sample split of the random forest model
    :param training_days_initial: number of trading days for the first training of the backtest
    :param test_days_step: number of trading days for testing
    :param threshold_probability_positive: probability threshold to consider a positive price prediction
    :return: None
    """

    model = RandomForestClassifier(n_estimators=estimators,  # number of trees: the higher, the better the accuracy
                                   min_samples_split=sample_split,  # the higher, the less accurate, but the less overfits
                                   random_state=1,  # if 1, same initialization
                                   n_jobs=-1)  # number of cores to be used (-1: max number of cores)

    # Train and backtest (train inside backtest)
    # NOTE: this is just a backtest to test all the possible successive trainings.
    # The deployed model will be trained using a given time range only.
    predictions = backtest(dataset, model, predictors_list,
                           days_initial_train=training_days_initial,
                           days_test=test_days_step,
                           threshold_probability_positive=threshold_probability_positive)

    # calculate precision score
    precision = precision_score(predictions["Target"], predictions["Predictions"])
    print("Precision to predict a positive day (?): ", precision, "\n")


def add_previous_behavior(data: pd.DataFrame, number_of_days_to_shift_list: list, predictors_list: list) -> tuple:
    """
    Add to the current day data from number_of_days_to_shift previous day. Data tags are defined in the predictors list.
    :param data: history of price
    :param number_of_days_to_shift_list: list of numbers of days in the past to associate the data of that day in the current day
    :param predictors_list: list of tags of data to be moved to the current day
    :return: dataframe with data of the past day and list of added column names
    """

    new_column_names_list = []
    for shift in number_of_days_to_shift_list:
        suffix = "-"+str(shift)
        for predictor in predictors_list:
            column_name = predictor+suffix
            data[column_name] = data.shift(shift)[predictor]
            new_column_names_list.append(column_name)
    return data, new_column_names_list

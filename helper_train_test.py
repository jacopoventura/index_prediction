import pandas as pd


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
    """
    all_predictions = []
    number_trading_days = data.shape[0]
    for i in range(days_initial_train, number_trading_days, days_test):
        train_dataset = data.iloc[:i].copy()
        test_dataset = data.iloc[i:(i+days_test)].copy()
        predictions = predict(train_dataset, test_dataset, predictors, model, threshold_probability_positive)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

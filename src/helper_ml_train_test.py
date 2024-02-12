import datetime
import yfinance as yf
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from stockstats import StockDataFrame


PRINT = False


def predict(train_data: pd.DataFrame, test_data: pd.DataFrame, predictors: list,
            model, threshold_probability_positive: float = .6) -> list:
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
    # a = model.predict_proba(test_data[predictors])
    # predictions = model.predict(test_data[predictors])  # predict the result (0 or 1)
    predictions = model.predict_proba(test_data[predictors])  # predict the probability of each possible class [0, 1]
    predicted_classes = check_prediction_probability_binary(predictions[:, 1], threshold_probability_positive)
    return predicted_classes


def check_prediction_probability_binary(prediction_probabilities: list, threshold_probability: float) -> list:
    """
    Apply the probability threshold to the probability vector (binary classification).
    :param prediction_probabilities: list of probabilities for one class (binary classification).
    :param threshold_probability: threshold
    :return: list of predicted classes based on the input probabilities
    """
    prediction_probabilities[prediction_probabilities >= threshold_probability] = 1
    prediction_probabilities[prediction_probabilities < threshold_probability] = 0
    return prediction_probabilities


def backtest(data: pd.DataFrame, model, predictors: list,
             days_initial_train: int = 2500, days_test: int = 250,
             threshold_probability_positive: float = .6) -> tuple:
    """
    Function to backtest the model.
    We train for the first days_initial_train days, and we test the following days_test days.
    Then we train for the (days_initial_train + k*days_test) days, and we test the following days_test days.
    NOTE: this is just a backtest to test all the possible successive trainings.
    """
    score_cumulative_train = {"Target": [], "Prediction": []}
    number_trading_days = data.shape[0]

    # Expand the training dataset
    for i in range(days_initial_train, number_trading_days, days_test):
        train_dataset = data.iloc[:i].copy()
        test_dataset = data.iloc[i:(i+days_test)].copy()
        # print("Train: ", train_dataset.index[0], " to ", train_dataset.index[-1], " days: ", train_dataset.shape[0])
        # print("Test : ", test_dataset.index[0], " to ", test_dataset.index[-1], " days: ", test_dataset.shape[0])
        predictions = predict(train_dataset, test_dataset, predictors, model, threshold_probability_positive)
        score_cumulative_train["Target"].append(test_dataset["Target"])
        score_cumulative_train["Prediction"].append(predictions)

    # Slide the training windows with a fixed size
    score_sliding_train = {"Target": [], "Prediction": []}
    k = 0
    for i in range(days_initial_train, number_trading_days, days_test):
        start_train_idx = k * days_test
        train_dataset = data.iloc[start_train_idx:i].copy()
        test_dataset = data.iloc[i:(i+days_test)].copy()
        # print("Train: ", train_dataset.index[0], " to ", train_dataset.index[-1], " days: ", train_dataset.shape[0])
        # print("Test : ", test_dataset.index[0], " to ", test_dataset.index[-1], " days: ", test_dataset.shape[0])
        predictions = predict(train_dataset, test_dataset, predictors, model, threshold_probability_positive)
        score_sliding_train["Target"].append(test_dataset["Target"])
        score_sliding_train["Prediction"].append(predictions)
        k += 1

    return score_cumulative_train, score_sliding_train

    # return pd.concat(all_predictions)


def train_and_deploy(data: pd.DataFrame, predictors: list,
                     start_date_training: datetime, end_date_training: datetime,
                     estimators: int = 200, sample_split: int = 50,
                     threshold_probability_positive: float = .6):
    """
    Train final model and save model parameters.
    :param data: full dataset
    :param predictors: list of predictors
    :param start_date_training: initial date for training
    :param end_date_training: final date for training
    :param estimators: number of trees for the forest
    :param sample_split: number of splits for the forest
    :param threshold_probability_positive: probability to accept a positive class as positive
    :return:
    """
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
    precision, specificity = compute_precision_specificity(test_dataset["Target"].tolist(), predicted_classes)
    print(f"Precision {precision:.2f},  specificity {specificity:.2f}")
    print(" ")


def create_and_test_random_forest(dataset: pd.DataFrame, predictors_list: list,
                                  estimators: int = 200, sample_split: int = 50,
                                  training_days_initial: int = 2500, test_days_step: int = 250,
                                  threshold_probability_positive: float = .6):
    """
    Create Random Forest model, train and backtest. A random forest classifier was chosen because it is resistant to overfit (due to the
    numerous random trees), runs quickly and handles non-linear relationships.
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
    accumulating_train, sliding_train = backtest(dataset, model, predictors_list,
                                                 days_initial_train=training_days_initial,
                                                 days_test=test_days_step,
                                                 threshold_probability_positive=threshold_probability_positive)

    # calculate scores
    # PRECISION: ability of the classifier not to label as positive a sample that is negative.
    # When the model predict a positive day, it was right precision% of times
    # SPECIFICITY: ability to predict a negative class correctly.
    # When the model predict a negative day, it was right specificity% of times
    if len(flatten(accumulating_train["Target"])) != len(flatten(accumulating_train["Prediction"])):
        print("ERROR: size error")
        exit()
    precision_cumulative, specificity_cumulative = compute_precision_specificity(flatten(accumulating_train["Target"]),
                                                                                 flatten(accumulating_train["Prediction"]))
    precision_sliding, specificity_sliding = compute_precision_specificity(flatten(sliding_train["Target"]),
                                                                           flatten(sliding_train["Prediction"]))
    print(f"Cumulative train: precision {precision_cumulative:.2f},  specificity {specificity_cumulative:.2f}")
    print(f"Sliding train   : precision {precision_sliding:.2f},  specificity {specificity_sliding:.2f}")
    print(" ")


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


def flatten(input_list: list) -> list:
    return [item for sub_list in input_list for item in sub_list]


def compute_precision_specificity(target_class: list, predicted_class: list) -> tuple:
    tn, fp, fn, tp = confusion_matrix(target_class, predicted_class, labels=[0, 1]).ravel()
    precision = 0.
    if (tp + fp) > 0:
        precision = tp / (tp + fp)
    specificity = 0.
    if (tn + fp) > 0:
        specificity = tn / (tn + fp)
    return precision, specificity


def query_and_prepare_dataset(ticker: str = "^GSPC",
                              prediction_target: str = "negative",
                              start_date: datetime = datetime.datetime(2000, 1, 1).date(),
                              previous_days_history: list = [5]):

    # STEP 1: query the historical data of the index from yahoo finance in OHLC ("Open-High-Low-Close") format
    price_history_df = yf.Ticker(ticker)
    price_history_df = price_history_df.history(period="max")

    # STEP 2: save the dataframe as StockDataFrame.
    # StockDataFrame is a particular dataframe for TA analysis of stocks.
    # A TA indicator can be simply calculated by adding a column name, like df["close_10_sma"] to calculate the 10 sma of the close.
    # Docu: https://github.com/jealous/stockstats/blob/master/README.md
    price_history_df = StockDataFrame.retype(price_history_df[["Open", "Close", "High", "Low", "Volume"]])

    # STEP 3: remove the hour from the date in order to merge dataframes of different tickers (ex: VIX)
    price_history_df["Trading date"] = [d.date() for d in price_history_df.index.to_list()]
    price_history_df.set_index("Trading date", inplace=True)

    # STEP 4: calculate the target to be predicted by the ML model.
    # Create a column with the closing price of the day after (which is the movement we want to predict)
    price_history_df["Tomorrow"] = price_history_df["close"].shift(-1)
    # Create the target column: if the next day is positive, then class 1 (positive class).
    price_history_df["Target"] = (price_history_df["Tomorrow"] > price_history_df["close"]).astype(int)
    # In binary classification, the class 1 is treated as the positive class to be predicted.
    # The positive class is the class we are most interested to predict.
    # If we are interested in predicting a negative day, we must label a negative next day with class 1 (positive)
    if prediction_target == "negative":
        price_history_df["Target"] = [1 - val for val in price_history_df["Target"]]  # negative day labelled with class 1

    # STEP 5: calculate Technical Analysis (TA) values
    # Average True Range
    price_history_df["atr"]

    # MACD
    price_history_df["macd"]

    # RSI
    periods_rsi = [5, 14]
    predictors_rsi = []
    for period in periods_rsi:
        column_name_rsi = f"rsi_{period}"
        price_history_df[column_name_rsi]
        predictors_rsi += [column_name_rsi]

    # SMA and ratios with SMA
    horizons_days_moving_average = [5, 50, 100]
    predictors_ma = []
    predictors_trend = []
    for horizon in horizons_days_moving_average:
        # Calculate the SMA
        ma_column_name = f"close_{horizon}_sma"
        price_history_df[ma_column_name]

        # add the column day close / MA close to the dataset
        column_name_ratio_with_ma = f"close_over_{horizon}sma"
        # Notably, the ma close is calculated using the current daily close, i.e. the enumerator
        price_history_df[column_name_ratio_with_ma] = price_history_df["close"] / price_history_df[ma_column_name]

        # calculate the trend: sum of positive days withing the selected horizon
        column_name_trend = f"Trend_{horizon}"
        # the shift(1) shifts the row by 1. This allows to calculate the rolling excluding the current day (row)
        price_history_df[column_name_trend] = price_history_df.shift(1).rolling(horizon).sum()["Target"] / horizon

        predictors_ma.append(column_name_ratio_with_ma)
        predictors_trend.append(column_name_trend)

    # Calculate the day of the year
    dates = price_history_df.index.to_list()
    price_history_df["Day of year"] = [d.timetuple().tm_yday / 365. for d in dates]

    # STEP 6: query the historical data of VIX in OHLC format
    vix = yf.Ticker("^VIX")
    vix = vix.history(period="max")
    vix = StockDataFrame.retype(vix[["Close"]])

    # Remove hour from index column (date)
    vix["Trading date"] = [d.date() for d in vix.index.to_list()]
    vix.set_index("Trading date", inplace=True)

    # TA data
    HORIZON_MA_VIX = 5
    column_name_sma_vix = f"close_{HORIZON_MA_VIX}_sma"
    vix[column_name_sma_vix]
    vix[column_name_sma_vix] = vix["close"] / vix[column_name_sma_vix]
    vix.rename(columns={"close": "vix", column_name_sma_vix: "vix/sma"}, inplace=True)

    # Concatenate VIX
    price_history_df = pd.concat([price_history_df, vix[["vix", "vix/sma"]]], axis=1)

    # Calculate daily price movement with respect to previous day close
    price_history_df_previous_day_close = price_history_df["close"].shift(1)
    price_history_df["Close/PDclose"] = price_history_df["close"] / price_history_df_previous_day_close
    price_history_df["Open/PDclose"] = price_history_df["open"] / price_history_df_previous_day_close
    price_history_df["High/PDclose"] = price_history_df["high"] / price_history_df_previous_day_close
    price_history_df["Low/PDclose"] = price_history_df["low"] / price_history_df_previous_day_close

    # Calculate VIX change with respect to previous day
    price_history_df["vix/PDvix"] = price_history_df["vix"] / price_history_df["vix"].shift(1)

    # Calculate Volume change with respect to previous day
    price_history_df["Volume/PDvolume"] = price_history_df["volume"] / price_history_df["volume"].shift(1)

    # # Utest
    # num_rows = price_history_df.shape[0]
    # sum_errors = 0
    # for i in range(1,num_rows-1):
    #     sum_errors += price_history_df["Close/prev day close"].iloc[i] - (price_history_df["Close"].iloc[i] / price_history_df["Close"].iloc[i-1])
    #     sum_errors += price_history_df["Open/prev day close"].iloc[i] - (price_history_df["Open"].iloc[i] / price_history_df["Close"].iloc[i - 1])
    #     sum_errors += price_history_df["High/prev day close"].iloc[i] - (price_history_df["High"].iloc[i] / price_history_df["Close"].iloc[i - 1])
    #     sum_errors += price_history_df["Low/prev day close"].iloc[i] - (price_history_df["Low"].iloc[i] / price_history_df["Close"].iloc[i - 1])
    # print(sum_errors)

    # Associate
    features_to_associate = ["Close/PDclose", "Open/PDclose"]
    price_history_df, past_features_associated_today_list = add_previous_behavior(price_history_df, previous_days_history, features_to_associate)

    # select specific interval
    price_history_df = price_history_df.loc[start_date:].copy()

    # store the last closed trading day
    last_closed_trading_day = price_history_df.iloc[-1]

    # remove rows containing at least 1 NaN
    price_history_df = price_history_df.dropna()

    predictors_dict = {"rsi": predictors_rsi,
                       "ma": predictors_ma,
                       "trend": predictors_trend,
                       "past features": past_features_associated_today_list}

    return price_history_df, predictors_dict, last_closed_trading_day

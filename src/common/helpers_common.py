import datetime
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from stockstats import StockDataFrame
from sklearn.preprocessing import StandardScaler


def split_dataset_with_index(dataset: pd.DataFrame, index_start_training: int, index_end_training: int, length_test_dataset: int) -> tuple:
    """
    Split dataset into training and test data according to their indexes.
    :param dataset: dataset
    :param index_start_training: index of the first data of the training subset
    :param index_end_training: index of the last data of the training subset
    :param length_test_dataset: length of the test dataset
    :return: train and test datasets
    """
    train_dataset = dataset.iloc[index_start_training:index_end_training].copy()
    test_dataset = dataset.iloc[index_end_training:(index_end_training + length_test_dataset)].copy()
    return train_dataset, test_dataset


def split_dataset_with_date(dataset: pd.DataFrame, start_date_training, end_date_training) -> tuple:
    """
        Split dataset into training and test data according to their indexes.
        :param dataset: dataset
        :param start_date_training: date of the first data of the training subset
        :param end_date_training: index of the last data of the training subset
        :return: train and test datasets
        """
    train_dataset = dataset.loc[start_date_training:end_date_training].copy()
    test_dataset = dataset.loc[end_date_training:].copy()
    return train_dataset, test_dataset


def compute_precision_recall_specificity(target_class: list, predicted_class: list) -> dict:
    """
    Compute precision, recall and specificity for a binary classifier.
    :param target_class: target class (list)
    :param predicted_class: predicted target class (list)
    :return: precision, recall, specificity, accuracy scores (dictionary)
    """
    tn, fp, fn, tp = confusion_matrix(target_class, predicted_class, labels=[0, 1]).ravel()
    print(f"FP: {fp}, FN: {fn}")
    # precision: capability to classify the positive class (false positive are important)
    precision = 0.
    if (tp + fp) > 0:
        precision = tp / (tp + fp)

    # recall (sensitivity): capability to classify the positive class (false negative are important)
    recall = 0.
    if (tp + fn) > 0:
        recall = tp / (tp + fn)

    # specificity: capability to classify the negative class
    specificity = 0.
    if (tn + fp) > 0:
        specificity = tn / (tn + fp)

    # accuracy
    accuracy = 0.
    if len(target_class)>0:
        accuracy = (tp + tn)/len(target_class)

    return {"precision": precision, "recall": recall, "specificity": specificity, "accuracy": accuracy}


def print_cumulative_sliding_scores(scores_cumulative: dict, scores_sliding: dict) -> None:
    """
    Print performance of the binary predictor for cumulative and sliding trainings.
    :param scores_cumulative: dictionary of scores in case of cumulative training
    :param scores_sliding: dictionary of scores in case of cumulative training
    :return None
    """
    precision = scores_cumulative["precision"]
    recall = scores_cumulative["recall"]
    specificity = scores_cumulative["specificity"]
    accuracy = scores_cumulative["accuracy"]
    print(f"Cumulative train: precision {precision:.2f},  "
          f"recall {recall:.2f}, specificity {specificity:.2f}, accuracy {accuracy:.2f}")
    precision = scores_sliding["precision"]
    recall = scores_sliding["recall"]
    specificity = scores_sliding["specificity"]
    accuracy = scores_sliding["accuracy"]
    print(f"Sliding train: precision {precision:.2f},  "
          f"recall {recall:.2f}, specificity {specificity:.2f}, accuracy {accuracy:.2f}")
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


def query_and_prepare_dataset(ticker: str = "^GSPC",
                              prediction_target: str = "negative",
                              horizon_days_prediction: int = 1,
                              start_date: datetime = datetime.datetime(2000, 1, 1).date(),
                              previous_days_history=None):
    if previous_days_history is None:
        previous_days_history = [5]
    HORIZON_MA_VIX = 10

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
    price_history_df["Tomorrow"] = price_history_df["close"].shift(-horizon_days_prediction)
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
    price_history_df["close/macd"] = price_history_df["close"] / price_history_df["macd"]

    # Aroon oscillator
    price_history_df["aroon"]
    price_history_df["aroon"] = price_history_df["aroon"] / 100

    # ADX
    price_history_df["adx"]

    # Bollinger band
    price_history_df["boll"]
    price_history_df["close/boll"] = price_history_df["close"] / price_history_df["boll"]

    # RSI
    periods_rsi = [5, 14]
    predictors_rsi = []
    for period in periods_rsi:
        column_name_rsi = f"rsi_{period}"
        price_history_df[column_name_rsi]
        predictors_rsi += [column_name_rsi]

    # SMA and ratios with SMA
    horizons_days_moving_average = [22, 50]  # [5, 50]
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
    features_to_associate = ["Close/PDclose"]
    price_history_df, past_features_associated_today_list = add_previous_behavior(price_history_df, previous_days_history, features_to_associate)

    # select specific interval
    price_history_df = price_history_df.loc[start_date:].copy()

    # store the last closed trading day
    now = datetime.datetime.now()
    if now.hour >= 22:
        last_closed_trading_day = price_history_df.iloc[-1]
    else:
        last_closed_trading_day = price_history_df.iloc[-2]

    # remove rows containing at least 1 NaN
    price_history_df = price_history_df.dropna()

    predictors_dict = {"rsi": predictors_rsi,
                       "ma": predictors_ma,
                       "trend": predictors_trend,
                       "past features": past_features_associated_today_list}

    price_history_df[price_history_df.select_dtypes(np.float64).columns] = price_history_df.select_dtypes(np.float64).astype(np.float32)
    price_history_df[price_history_df.select_dtypes(np.int64).columns] = price_history_df.select_dtypes(np.int64).astype(np.float32)

    #scaler = StandardScaler()
    #columns_to_scale = price_history_df.columns[price_history_df.columns != "Target"]
    #price_history_df[columns_to_scale] = scaler.fit_transform(price_history_df[columns_to_scale])

    return price_history_df, predictors_dict, last_closed_trading_day


def check_prediction_probability_binary(prediction_probabilities: list, threshold_probability: float) -> list:
    """
    Apply the probability threshold to the probability vector (binary classification).
    :param prediction_probabilities: list of probabilities for one class (binary classification).
    :param threshold_probability: threshold
    :return: list of predicted classes based on the input probabilities
    """
    predicted_class = [1 if prob >= threshold_probability else 0 for prob in prediction_probabilities]
    return predicted_class


def stats_cumulative_sliding_train(cumulative_training, sliding_training):
    scores_cumulative = compute_precision_recall_specificity(flatten(cumulative_training["Target"]),
                                                             flatten(cumulative_training["Prediction"]))
    scores_sliding = compute_precision_recall_specificity(flatten(sliding_training["Target"]),
                                                          flatten(sliding_training["Prediction"]))
    print_cumulative_sliding_scores(scores_cumulative, scores_sliding)
    scores_cumulative_last = compute_precision_recall_specificity(cumulative_training["Target"][-1],
                                                                  cumulative_training["Prediction"][-1])
    scores_sliding_last = compute_precision_recall_specificity(sliding_training["Target"][-1],
                                                               sliding_training["Prediction"][-1])
    precision = scores_cumulative_last["precision"]
    recall = scores_cumulative_last["recall"]
    specificity = scores_cumulative_last["specificity"]
    accuracy = scores_cumulative_last["accuracy"]
    print(f"Last cumulative train: precision {precision:.2f},  "
          f"recall {recall:.2f}, specificity {specificity:.2f}, accuracy {accuracy:.2f}")
    precision = scores_sliding_last["precision"]
    recall = scores_sliding_last["recall"]
    specificity = scores_sliding_last["specificity"]
    accuracy = scores_sliding_last["accuracy"]
    print(f"Last sliding train: precision {precision:.2f},  "
          f"recall {recall:.2f}, specificity {specificity:.2f}, accuracy {accuracy:.2f}")



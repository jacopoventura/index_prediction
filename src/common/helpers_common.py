# Copyright (c) 2024 Jacopo Ventura

import datetime
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from stockstats import StockDataFrame
from sklearn import preprocessing
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)


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
    if len(target_class) > 0:
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


def print_classification_scores(scores_dict: dict) -> None:
    """Print classification scores.
    :param: dictionary of scores
    """
    precision = scores_dict["precision"]
    recall = scores_dict["recall"]
    specificity = scores_dict["specificity"]
    accuracy = scores_dict["accuracy"]
    print(f"Test data: precision {precision:.2f},  "
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


def query_data(ticker: str = "SPY", start_date=None) -> pd.DataFrame:
    """
    Query stock price daily data.
    :param ticker: ticker of the security
    :param start_date: initial date of the database
    :return: price history (High, Low, Open, Close, Volume) in pd.Dataframe format
    """

    # STEP 1: query the dataset
    df = yf.Ticker(ticker)
    if start_date is None:
        df = df.history(period="max")
    else:
        df = df.history(start=start_date)
    if "Dividends" in df.columns:
        df.drop(columns="Dividends", inplace=True)
    if "Stock Splits" in df.columns:
        df.drop(columns="Stock Splits", inplace=True)
    if "Capital Gains" in df.columns:
        df.drop(columns="Capital Gains", inplace=True)

    # STEP 2: remove the hour from the date in order to merge dataframes of different tickers (ex: VIX)
    df["Trading date"] = [d.date() for d in df.index.to_list()]
    df.set_index("Trading date", inplace=True)

    # STEP 3: reduce data type to float32
    df[df.select_dtypes(np.float64).columns] = df.select_dtypes(np.float64).astype(np.float32)
    df[df.select_dtypes(np.int64).columns] = df.select_dtypes(np.int64).astype(np.float32)

    return df


def calculate_technical_analysis(df: pd.DataFrame,
                                 start_date=None,
                                 add_vix: bool = False,
                                 ratio_close_over_indicator: bool = False,
                                 ) -> tuple:
    """
    Calculates the technical analysis values for each day of the database.
    The following technical analysis values are calculated:
    atr, macd, close/macd, aroon, adx, boll, close/boll, rsi, sma, close/sma,
    close/Previous Day (PD) close, open/PDclose, high/PDclose, low/PDclose
    day of the year
    vix, sma of vix, vix / sma of vix, vix / previous day vix
    NOTE: the first day of the returned dataframe is not as the input first day because
    some of the TA indicators are NaN in the first days.
    :param df:
    :param start_date:
    :param add_vix:
    :param ratio_close_over_indicator: if True, compute the ratio close / indicator
    :return:
    """

    # Parameters for technical analysis
    periods_rsi = [5, 14]
    horizons_days_moving_average = [22, 50]  # [5, 50]

    # STEP 1: convert the DataFrame to a StockDataFrame for easy technical analysis calculation
    # StockDataFrame is a particular dataframe for TA analysis of stocks.
    # A TA indicator can be simply calculated by adding a column name, like df["close_10_sma"] to calculate the 10 sma of the close.
    # Docu: https://github.com/jealous/stockstats/blob/master/README.md
    # Question: why MA are not NaN in the first day?
    df_stock_format = StockDataFrame.retype(df)

    # STEP 2: calculate Technical Analysis (TA) values
    # save all the new column names in a dictionary
    ta_features_dict = {"hloc": ["high", "low", "open", "close"],
                        "volume": ["volume"]}

    # Average True Range
    df_stock_format["atr"]
    ta_features_dict["atr"] = ["atr"]

    # MACD
    df_stock_format["macd"]
    ta_features_dict["macd"] = ["macd"]
    if ratio_close_over_indicator:
        df_stock_format["close/macd"] = df_stock_format["close"] / df_stock_format["macd"]
        ta_features_dict["macd"].append("close/macd")

    # Aroon oscillator
    df_stock_format["aroon"]
    ta_features_dict["aroon"] = ["aroon"]

    # ADX
    df_stock_format["adx"]
    ta_features_dict["adx"] = ["adx"]

    # Bollinger band
    df_stock_format["boll"]
    ta_features_dict["bollinger"] = ["boll", "boll_ub", "boll_lb"]
    if ratio_close_over_indicator:
        df_stock_format["close/boll"] = df_stock_format["close"] / df_stock_format["boll"]
        df_stock_format["close/boll_ub"] = df_stock_format["close"] / df_stock_format["boll_ub"]
        df_stock_format["close/boll_lb"] = df_stock_format["close"] / df_stock_format["boll_lb"]
        ta_features_dict["bollinger"].append("close/boll")
        ta_features_dict["bollinger"].append("close/boll_ub")
        ta_features_dict["bollinger"].append("close/boll_lb")

    # RSI
    ta_features_dict["rsi"] = []
    for period in periods_rsi:
        column_name_rsi = f"rsi_{period}"
        df_stock_format[column_name_rsi]
        ta_features_dict["rsi"].append(column_name_rsi)

    # SMA and ratios with SMA
    ta_features_dict["ma"] = []
    for horizon in horizons_days_moving_average:
        # Calculate the SMA
        ma_column_name = f"close_{horizon}_sma"
        df_stock_format[ma_column_name]
        ta_features_dict["ma"].append(ma_column_name)

        # add the column day close / MA close to the dataset
        if ratio_close_over_indicator:
            column_name_ratio_with_ma = f"close/{horizon}sma"
            # Notably, the ma close is calculated using the current daily close, i.e. the enumerator
            df_stock_format[column_name_ratio_with_ma] = df_stock_format["close"] / df_stock_format[ma_column_name]
            ta_features_dict["ma"].append(column_name_ratio_with_ma)

    # Day of the year
    dates = df_stock_format.index.to_list()
    df_stock_format["day_of_year"] = [d.timetuple().tm_yday / 365. for d in dates]
    ta_features_dict["day"] = ["day_of_year"]

    # Calculate daily price movement with respect to previous day close
    if ratio_close_over_indicator:
        price_history_df_previous_day_close = df_stock_format["close"].shift(1)
        df_stock_format["close/PDclose"] = df_stock_format["close"] / price_history_df_previous_day_close
        df_stock_format["open/PDclose"] = df_stock_format["open"] / price_history_df_previous_day_close
        df_stock_format["high/PDclose"] = df_stock_format["high"] / price_history_df_previous_day_close
        df_stock_format["low/PDclose"] = df_stock_format["low"] / price_history_df_previous_day_close
        ta_features_dict["price/previous day"] = ["close/PDclose", "open/PDclose", "high/PDclose", "low/PDclose"]

    # STEP 3: query the historical data of VIX in OHLC format
    HORIZON_MA_VIX = 10
    if add_vix:
        vix_df = query_data("^VIX", start_date=start_date)
        vix_df_stock_format = StockDataFrame.retype(vix_df[["Close"]])

        # TA data
        column_name_sma_vix = f"close_{HORIZON_MA_VIX}_sma"
        vix_df_stock_format[column_name_sma_vix]
        vix_df_stock_format.rename(columns={"close": "vix", column_name_sma_vix: "vix_sma"}, inplace=True)
        ta_features_dict["vix"] = ["vix", "vix_sma"]
        if ratio_close_over_indicator:
            vix_df_stock_format["vix/sma"] = vix_df_stock_format["vix"] / vix_df_stock_format["vix_sma"]
            ta_features_dict["vix"].append("vix/sma")

        # Concatenate VIX
        first_day_price = df_stock_format.index[0]
        first_day_vix = vix_df_stock_format.index[0]
        if first_day_price >= first_day_vix:
            df_stock_format = pd.concat([df_stock_format, vix_df_stock_format.loc[first_day_price:]], axis=1)
        else:
            df_stock_format = pd.concat([df_stock_format.loc[first_day_vix:], vix_df_stock_format], axis=1)

        # Calculate VIX change with respect to previous day
        if ratio_close_over_indicator:
            df_stock_format["vix/PDvix"] = df_stock_format["vix"] / df_stock_format["vix"].shift(1)
            ta_features_dict["vix"].append("vix/PDvix")

    # STEP 4: remove rows with NaNs
    # (in the first days, some indicators like MA, macd are not available, thus they are NaN)
    df_stock_format.dropna(inplace=True)

    # Calculate Volume change with respect to previous day
    # df_stock_format["Volume/PDvolume"] = df_stock_format["volume"] / df_stock_format["volume"].shift(1)

    # # Utest
    # num_rows = price_history_df.shape[0]
    # sum_errors = 0
    # for i in range(1,num_rows-1):
    #     sum_errors += price_history_df["Close/prev day close"].iloc[i] - (price_history_df["Close"].iloc[i] / price_history_df["Close"].iloc[i-1])
    #     sum_errors += price_history_df["Open/prev day close"].iloc[i] - (price_history_df["Open"].iloc[i] / price_history_df["Close"].iloc[i - 1])
    #     sum_errors += price_history_df["High/prev day close"].iloc[i] - (price_history_df["High"].iloc[i] / price_history_df["Close"].iloc[i - 1])
    #     sum_errors += price_history_df["Low/prev day close"].iloc[i] - (price_history_df["Low"].iloc[i] / price_history_df["Close"].iloc[i - 1])
    # print(sum_errors)

    # select specific interval
    if start_date is not None:
        df_stock_format = df_stock_format.loc[start_date:].copy()

    return pd.DataFrame(df_stock_format), ta_features_dict


def scale_dataset(df, scale: bool = True) -> dict:
    """
    Scale the dataframe.
    Note: dataframe is a mutable object, thus is passed by reference (assignment).
    :param df: dataframe of the price history with technical analysis
    :param scale: boolean flag to scale the data or not.
    :return: dictionary of scaler.
    """
    column_scaler = {}
    if scale:
        # scale the data (prices) from 0 to 1
        for column in df.columns:
            if column != "future":
                scaler = preprocessing.MinMaxScaler()
                df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
                column_scaler[column] = scaler
        assert "future" in df.columns
        df["future"] = column_scaler["close"].transform(np.expand_dims(df["future"].values, axis=1))
    return column_scaler


def add_date_column(df):
    if "date" not in df.columns:
        df["date"] = df.index


def check_last_day_of_dataset(df: pd.DataFrame) -> None:
    """
    Check if the last day of the dataset was queried during a trading session.
    If so, remove it.
    :param df: dataset of price history
    :return: None
    """
    now = datetime.datetime.now()
    if now.hour < 22 and now.day == df.iloc[-1].name.day:
        # data of the last day were queried during a trading session, thus remove last row
        df.drop(index=df.index[-1], axis=0, inplace=True)


def calc_price_change(price_start: float, price_end: float) -> float:
    return 100. * (price_end - price_start) / price_start


def calc_change_category(pct: float) -> int:
    if pct >= 0:
        return 1
    elif 0 > pct >= -1:
        return 2
    elif -1 > pct >= -2:
        return 3
    else:
        return 4


def calc_change_category_back(category: int) -> tuple:
    """
    Given the input change category, return the price change interval as tuple.
    :param category: price change category
    :return: interval of price change (pct)
    """
    match category:
        case 1:
            return 1, 0
        case 2:
            return 0, -1
        case 3:
            return -1, -2
        case 4:
            return -2, -3


def calc_performance_category(category_test: list, category_predicted: list) -> None:
    """
    Calculate the performance in terms of price change category.
    :param category_test: categories of test data
    :param category_predicted: categories of predicted data
    """
    # Script can be improved by getting categories from a set.
    num_tests = len(category_test)
    assert num_tests == len(category_predicted)
    category_1_match = 0
    category_2_match = 0
    category_2_negative = 0
    category_3_match = 0
    category_3_negative = 0
    category_4_match = 0
    category_4_negative = 0
    count_negative_match = 0
    num_cat_1_test = len(np.where(np.array(category_test) == 1)[0])
    num_cat_2_test = len(np.where(np.array(category_test) == 2)[0])
    num_cat_3_test = len(np.where(np.array(category_test) == 3)[0])
    num_cat_4_test = len(np.where(np.array(category_test) == 4)[0])
    num_cat_1_pred = len(np.where(np.array(category_predicted) == 1)[0])
    # num_cat_2_pred = len(np.where(np.array(category_predicted) == 2)[0])
    # num_cat_3_pred = len(np.where(np.array(category_predicted) == 3)[0])
    # num_cat_4_pred = len(np.where(np.array(category_predicted) == 4)[0])
    for idx, category in enumerate(category_test):
        if category > 1 and category_predicted[idx] > 1:
            count_negative_match += 1
        if category == 1:
            if category_predicted[idx] == category:
                category_1_match += 1
        if category == 2:
            if category_predicted[idx] == category:
                category_2_match += 1
            elif category_predicted[idx] > 1:  # still negative prediction
                category_2_negative += 1
        if category == 3:
            if category_predicted[idx] == category:
                category_3_match += 1
            elif category_predicted[idx] > 1:  # still negative prediction
                category_3_negative += 1
        if category == 4:
            if category_predicted[idx] == category:
                category_4_match += 1
            elif category_predicted[idx] > 1:  # still negative prediction
                category_4_negative += 1

    match_rate_1 = 0
    if num_cat_1_test > 0:
        match_rate_1 = category_1_match / num_cat_1_test

    match_rate_2 = 0
    negative_rate_2 = 0
    if num_cat_2_test > 0:
        match_rate_2 = category_2_match / num_cat_2_test
        negative_rate_2 = category_2_negative / num_cat_2_test

    match_rate_3 = 0
    negative_rate_3 = 0
    if num_cat_3_test > 0:
        match_rate_3 = category_3_match / num_cat_3_test
        negative_rate_3 = category_3_negative / num_cat_3_test

    match_rate_4 = 0
    negative_rate_4 = 0
    if num_cat_4_test > 0:
        match_rate_4 = category_4_match / num_cat_4_test
        negative_rate_4 = category_4_negative / num_cat_4_test

    count_negative_test = num_tests - num_cat_1_test
    count_negative_prediction = num_tests - num_cat_1_pred
    negative_match_rate = count_negative_match / count_negative_test

    category_performance_dict = {
        "negative": {"match rate": negative_match_rate},
        "1": {"match rate": match_rate_1},
        "2": {"match rate": match_rate_2, "negative": negative_rate_2},
        "3": {"match rate": match_rate_3, "negative": negative_rate_3},
        "4": {"match rate": match_rate_4, "negative": negative_rate_4},
    }

    print_performance_category(category_performance_dict)


def print_performance_category(performance_dict) -> None:
    match_1 = performance_dict["1"]["match rate"]
    match_2 = performance_dict["2"]["match rate"]
    match_3 = performance_dict["3"]["match rate"]
    match_4 = performance_dict["4"]["match rate"]
    match_negative = performance_dict["negative"]["match rate"]
    negative_2 = performance_dict["2"]["negative"]
    negative_3 = performance_dict["3"]["negative"]
    negative_4 = performance_dict["4"]["negative"]
    print(f"Positive change match: {match_1:.2f}")
    print(f"Negative change match: {match_negative:.2f}")
    print(f"[0; -1]% change match: {match_2:.2f}")  # , negative prediction rate: {negative_2:.2f}")
    print(f"[-1; -2]% change match: {match_3:.2f}")  # , negative prediction rate: {negative_3:.2f}")
    print(f"[-2; -3]% change match: {match_4:.2f}")  # , negative prediction rate: {negative_4:.2f}")
    print(" ")


def add_target_price_to_predict(df: pd.DataFrame, lookup_step: int = 1) -> None:
    """
    Add a new column to the dataframe with the price to predict.
    Note: dataframe is a mutable object, thus is passed by reference (assignment).
    :param df: dataframe of the price history data
    :param lookup_step: prediction horizon
    :return: None
    """
    # add the target column (label) by shifting by `lookup_step`
    df['future'] = df['close'].shift(-lookup_step)


def calc_relative_change_with_levels(start_price_list: np.array, end_price_list: np.array):
    # Create the target column for classification: if the next day is positive, then class 1 (positive class).

    assert np.size(start_price_list) == np.size(end_price_list)
    len_data = np.size(start_price_list)
    if len_data > 1:
        is_positive_change = [int(end_price_list[i] > start_price_list[i]) for i in range(len_data)]
        pct_change = [calc_price_change(start_price_list[i], end_price_list[i]) for i in range(len_data)]
        category_pct_change = [calc_change_category(pct) for pct in pct_change]
    else:
        start_price_list = float(start_price_list[0])
        end_price_list = float(end_price_list[0])
        is_positive_change = [1 if end_price_list > start_price_list else 0]
        pct_change = [calc_price_change(start_price_list, end_price_list)]
        category_pct_change = [calc_change_category(pct_change[0])]

    return is_positive_change, pct_change, category_pct_change


def extract_close_from_sequence(last_sequence: np.array, feature_list: list) -> np.array:
    """
    Extract the cloe price of the latest day of the last_sequence.
    This closing price is the close with respect to which the ML model shall predict the future price.
    :param last_sequence: array with the last sequence
    :param feature_list: list of feature names (columns)
    :return: list of closing prices
    """
    assert "close" in feature_list
    idx_close = feature_list.index("close")
    last_sequence_close = [0] * last_sequence.shape[0]
    for idx, sequence in enumerate(last_sequence):
        last_sequence_close[idx] = sequence[-1][idx_close]
    return last_sequence_close


def get_last_sequence_for_prediction(df: pd.DataFrame, feature_list: list,
                                     lookup_step: int = 1,
                                     number_previous_days: int = 50, lstm=True) -> dict:
    """
    Get the last sequence of data to apply the model for prediction.
    The last sequence has size (lookup_step, number_previous_days, num_features) in case of lstm,
    (lookup_step, num_features) in case of classical neural network / Machine learning.
    In fact, the last day of the dataset is the last closed trading day.
    If the lookup_step > 1, the model will predict the future price after lookup_step from the last day.
    If we want to predict the price of the day after the last day of the dataset,
    we need the (last - lookup_step) day.
    :param df: dataframe of the price history data with the technical analysis
    :param feature_list: list of feature names (columns of the dataframe)
    :param lookup_step: number of days in the future for prediction
    :param number_previous_days: number of previous days for LSTM
    :param lstm: bool for data for lstm model
    :return: last sequence of data and last sequence of close and date close (as dict)
    """

    # extract last data for prediction and remove last trading day
    if lstm:
        # prepare data for LSTM model
        # extract the last number_previous_days rows from the dataset
        # add_date_column(df)
        last_sequence = create_data_format_for_lstm(df.iloc[-(number_previous_days+lookup_step-1):],
                                                    feature_list=feature_list,
                                                    previous_days_history=number_previous_days)[0]
        # last_sequence = np.array(df[feature_list].tail(number_previous_days))
        # expands the shape from (number_previous_days, num_features) to (1, number_previous_days, num_features)
        # last_sequence = np.expand_dims(last_sequence, axis=0)
        last_sequence_close = extract_close_from_sequence(last_sequence, feature_list)
        last_sequence_date_close = df.tail(lookup_step).index.values.tolist()
    else:
        # prepare data for not LSTM model
        # The last sequence consists of the last lookup_step rows
        # In fact, given the last closed trading day in the dataset, if we want the price of tomorrow,
        # we need the data of (tomorrow - lookup_step) days.
        last_sequence = df.tail(lookup_step).copy()
        assert "close" in feature_list
        last_sequence_close = last_sequence["close"]
        last_sequence_date_close = last_sequence.index.to_list()
        # remove the last sequence from dataset
    df.drop(df.tail(lookup_step).index, inplace=True)

    # create output
    last_sequence_dict = {"sequence": last_sequence,
                          "close": last_sequence_close,
                          "date close": last_sequence_date_close}

    return last_sequence_dict


def query_and_calculate_ta_features(ticker: str = "^GSPC",
                                    start_date_database=None,
                                    add_vix: bool = False,
                                    calculate_ratio_close_with_indicators: bool = True) -> tuple:
    """
    Query price history data and calculate the technical analysis.
    :param ticker: ticker of the security
    :param start_date_database: initial date of the dataset
    :param add_vix: flag to use the vix data
    :param calculate_ratio_close_with_indicators: flag to compute the ratio close / indicator
    :return: pd.DataFrame with price data and the TA indicators, column scaler dict and list of column features
    """

    # STEP 1: query data and store them in a float32 DataFrame
    price_history_df = query_data(ticker, start_date=start_date_database)

    # STEP 2: calculate technical analysis
    price_history_df_with_ta, predictors_dict = calculate_technical_analysis(price_history_df,
                                                                             start_date=start_date_database,
                                                                             add_vix=add_vix,
                                                                             ratio_close_over_indicator=calculate_ratio_close_with_indicators)

    return price_history_df_with_ta, predictors_dict


def prepare_data_for_ml(price_history_df_with_ta: pd.DataFrame,
                        feature_list: list,
                        horizon_days_prediction: int = 1,
                        previous_days_history: int = 50,
                        scale: bool = True,
                        lstm: bool = True) -> tuple:
    """
    Prepare data for the machine learning model.
    The last sequence has size (lookup_step, number_previous_days, num_features) in case of lstm,
    (lookup_step, num_features) in case of classical neural network / Machine learning.
    In fact, the last day of the dataset is the last closed trading day.
    If the lookup_step > 1, the model will predict the future price after lookup_step from the last day.
    If we want to predict the price of the day after the last day of the dataset,
    we need the (last - lookup_step) day.
    :param price_history_df_with_ta: dataset with the price history
    :param feature_list: list of feature names (column names of the dataframe)
    :param horizon_days_prediction: number of days for the price prediction
    :param previous_days_history: number of past days to consider for price prediction (lstm)
    :param scale: flag to scale the data
    :param lstm: flag to use lstm
    :return: last sequence, X, y, column_scaler
    """

    # STEP 1: remove the last day in the dataset if queried during a trading day
    # We remove it before calculating the target because it has a close value,
    # which will be considered as target for the -horizon_days_prediction day (which is incorrect).
    check_last_day_of_dataset(price_history_df_with_ta)

    # STEP 2: calculate target to predict (new column in the dataframe)
    add_target_price_to_predict(price_history_df_with_ta, lookup_step=horizon_days_prediction)

    is_positive_change, pct_change, category_pct_change = calc_relative_change_with_levels(price_history_df_with_ta["close"].values,
                                                                                           price_history_df_with_ta["future"].values)

    # STEP 3: scale data
    column_scaler = scale_dataset(price_history_df_with_ta, scale=scale)

    # STEP 4: add change % columns to dataframe (after scaling because we don't want to scale them)
    price_history_df_with_ta["change %"] = pct_change
    price_history_df_with_ta["is positive change"] = is_positive_change
    price_history_df_with_ta["category % change"] = category_pct_change

    # STEP 5: get the last sequence for actual prediction
    last_sequence_dict = get_last_sequence_for_prediction(price_history_df_with_ta,
                                                          feature_list=feature_list,
                                                          lookup_step=horizon_days_prediction,
                                                          number_previous_days=previous_days_history,
                                                          lstm=lstm)

    # STEP 6: switch to data format for LSTM
    if lstm:
        X, y = create_data_format_for_lstm(price_history_df_with_ta,
                                           feature_list=feature_list,
                                           previous_days_history=previous_days_history)
    else:
        X = price_history_df_with_ta[feature_list].values
        y = price_history_df_with_ta["target"].values

    return last_sequence_dict, X, y, column_scaler


def get_X_y_from_df(df_array: np.array, target_array: np.array, previous_days_history: int) -> tuple:
    # date_list = df.index.to_list()
    sequence_data = [[np.array(df_array[i - previous_days_history + 1:i + 1]), target_array[i]] for i in
                     range(previous_days_history - 1, len(df_array))]
    # construct the X's and y's
    X, y = [], []
    for sequence, target in sequence_data:
        X.append(sequence)
        y.append(target)
    # convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    return X, y


def create_data_format_for_lstm(df: pd.DataFrame, feature_list: list, previous_days_history: int) -> tuple:
    # List comprehension
    # List of tuples ( [n_steps, n_features], target), where [n_steps, n_features] is the input for the LSTM
    # NOTE: using iloc is 20x slower, thus we create a copy of df.values
    # df_array = df[feature_list+["date"]].values.copy()
    df["date"] = df.index.to_list()
    df_array = df[feature_list+["date"]].values.copy()
    target_array = df['future'].values.copy()
    X, y = get_X_y_from_df(df_array, target_array, previous_days_history)
    X = X[:, :, :len(feature_list)].astype(np.float32)
    return X, y


def query_and_prepare_dataset(ticker: str = "^GSPC",
                              prediction_target: str = "negative",
                              horizon_days_prediction: int = 1,
                              start_date: datetime = datetime.datetime(2000, 1, 1).date(),
                              previous_days_history=None,
                              vix=True):
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
    if vix:
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

        # Calculate VIX change with respect to previous day
        price_history_df["vix/PDvix"] = price_history_df["vix"] / price_history_df["vix"].shift(1)

    # Calculate daily price movement with respect to previous day close
    price_history_df_previous_day_close = price_history_df["close"].shift(1)
    price_history_df["Close/PDclose"] = price_history_df["close"] / price_history_df_previous_day_close
    price_history_df["Open/PDclose"] = price_history_df["open"] / price_history_df_previous_day_close
    price_history_df["High/PDclose"] = price_history_df["high"] / price_history_df_previous_day_close
    price_history_df["Low/PDclose"] = price_history_df["low"] / price_history_df_previous_day_close

    # Calculate Volume change with respect to previous day
    # price_history_df["Volume/PDvolume"] = price_history_df["volume"] / price_history_df["volume"].shift(1)

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
    if previous_days_history is not None:
        price_history_df, past_features_associated_today_list = add_previous_behavior(price_history_df, previous_days_history, features_to_associate)
    else:
        past_features_associated_today_list = []

    # select specific interval
    if start_date is not None:
        price_history_df = price_history_df.loc[start_date:].copy()

    # store the last closed trading day
    now = datetime.datetime.now()
    if now.hour >= 22:
        last_closed_trading_day = price_history_df.iloc[-1]
    else:
        last_closed_trading_day = price_history_df.iloc[-2]
        # remove last row

    # remove rows containing at least 1 NaN
    # price_history_df = price_history_df.dropna()

    predictors_dict = {"rsi": predictors_rsi,
                       "ma": predictors_ma,
                       "trend": predictors_trend,
                       "past features": past_features_associated_today_list}

    price_history_df[price_history_df.select_dtypes(np.float64).columns] = price_history_df.select_dtypes(np.float64).astype(np.float32)
    price_history_df[price_history_df.select_dtypes(np.int64).columns] = price_history_df.select_dtypes(np.int64).astype(np.float32)

    # scaler = StandardScaler()
    # columns_to_scale = price_history_df.columns[price_history_df.columns != "Target"]
    # price_history_df[columns_to_scale] = scaler.fit_transform(price_history_df[columns_to_scale])

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


def behavior_of_the_price(df: pd.DataFrame) -> None:
    print("============================== behavior of the price =================================")
    print("Number of trading days in the dataset: ", df.shape[0])
    number_of_days = df.shape[0]
    if "Target" in df.columns:
        percentage_positive_days = df["Target"].sum() / number_of_days
    else:
        percentage_positive_days = df["is positive change"].sum() / number_of_days
    percentage_negative_days = 1 - percentage_positive_days
    print("Number of trading days: ", number_of_days)
    print(f"Percentage of positive days: {percentage_positive_days:.2f}")
    print(f"Percentage of negative days: {percentage_negative_days:.2f}")
    print(" ")
    # Old version: exclusion of the training dataset
    # number_of_days_tested = df.iloc[num_days_train:].shape[0]
    # percentage_positive_days = df["Target"].iloc[num_days_train:].sum() / number_of_days_tested
    # percentage_negative_days = 1 - percentage_positive_days
    # print("Number of trading days used to test the model: ", number_of_days_tested)
    # print(f"Percentage of positive days in the dataset: {percentage_positive_days:.2f}")
    # print(f"Percentage of negative days in the dataset: {percentage_negative_days:.2f}")
    # print(" ")


def plot_test_price_prediction(Y_predicted: list, Y_test: list) -> None:
    """
    Plot price prediction of the test phase.
    :param Y_predicted: list of predicted price values
    :param Y_test: list of expected price values
    :return: None
    """
    plt.plot(Y_test, c='b')
    plt.plot(Y_predicted, c='r')
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend(["Actual Price", "Predicted Price"])
    plt.title("TEST of the model")
    plt.show()


def plot_price_prediction(predicted_price: list, price_close_test_data: list, last_sequence_close: list) -> None:
    # Plot predictions
    NUM_DAYS_IN_PLOT = 10
    num_days_from_test = NUM_DAYS_IN_PLOT - len(last_sequence_close)
    if num_days_from_test < 0:
        num_days_from_test = 0
        y_data_plot = last_sequence_close
        x_data_plot = range(len(last_sequence_close))
    else:
        y_data_plot = np.append(price_close_test_data[-num_days_from_test:], last_sequence_close)
        x_data_plot = range(len(last_sequence_close) + num_days_from_test)

    y_future = np.append(y_data_plot[-1], predicted_price)
    x_future = range(max(x_data_plot), max(x_data_plot) + len(predicted_price) + 1)
    plt.plot(x_data_plot, y_data_plot, c='b')
    plt.plot(x_future, y_future, c = 'r')
    # plt.plot(y_predicted, c='r')
    plt.xlabel("Day")
    plt.ylabel("Price")
    plt.legend(["Actual Price", "Predicted Price"])
    plt.title("PREDICTION")
    plt.show()
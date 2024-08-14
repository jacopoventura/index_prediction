# Copyright (c) 2024 Jacopo Ventura

import math
import pickle
import os
import tensorflow as tf
from src.common.helpers_common import *
import streamlit as st
import sys
import warnings


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)
warnings.filterwarnings("ignore")

# TODO:
# 1. write readme
# 3. remove icons
# 4. update portfolio

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    st.set_page_config(layout="wide")

    # App title
    st.markdown("<h1 style='text-align: center; '>Stock price prediction</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; '>with Machine Learning</h4>", unsafe_allow_html=True)

    # inputs
    TICKER = st.selectbox(
        "ticker",
        ("SPY", ),
    )
    prediction_horizon = st.selectbox(
        "predict",
        ("next day close", ),
    )

    LOOKUP_STEP = 1
    if prediction_horizon == "next day close":
        LOOKUP_STEP = 1

    TYPE_OF_PREDICTION = st.selectbox(
        "type of prediction",
        ("price", "change"),
    )

    MODEL_FILENAME = "LSTM_" + TICKER + f"_lookup_{LOOKUP_STEP}"
    MODEL_PATH = os.path.join("trained models", MODEL_FILENAME)

    # ======================================================================================
    #                                    LOAD THE TRAINED MODEL
    # ======================================================================================

    # ML model
    trained_model = tf.saved_model.load(MODEL_PATH)

    # model parameters
    with open(MODEL_PATH + "/parameters_dict.pkl", 'rb') as f:
        parameters_dict = pickle.load(f)

    BIAS = parameters_dict["BIAS"]
    final_bias = parameters_dict["FINAL_BIAS"]
    ADD_VIX = parameters_dict["ADD_VIX"]
    CALCULATE_RATIOS = parameters_dict["CALCULATE_RATIOS"]
    LSTM = parameters_dict["LSTM"]
    PREVIOUS_DAYS_HISTORY = parameters_dict["PREVIOUS_DAYS_HISTORY"]
    SELECTED_FEATURES = parameters_dict["SELECTED_FEATURES"]
    SCALE = parameters_dict["SCALE"]

    # column scaler
    with open(MODEL_PATH + "/scaler.pkl", 'rb') as f:
        column_scaler_loaded = pickle.load(f)

    # ======================================================================================
    #                                    QUERY AND PREPARE DATA
    # ======================================================================================

    # Calculate initial date to query the price history (avoid querying unnecessary data) based on the required history length
    number_of_weeks = math.ceil(PREVIOUS_DAYS_HISTORY / 5) + 1  # history is defined as past trading days, thus no holiday / weekends
    number_of_total_days = number_of_weeks * 7 + 4  # add few days as buffer for bank holidays

    INITIAL_DATE_OF_DATASET = datetime.datetime.today().date() - datetime.timedelta(number_of_total_days)

    # if the start date is a weekend day, shift additional 2 days
    if INITIAL_DATE_OF_DATASET.weekday() > 4:
        INITIAL_DATE_OF_DATASET -= datetime.timedelta(2)  # go back by one weekend

    # query price history and calculate TA
    price_history_df, predictors_dict = query_and_calculate_ta_features(ticker=TICKER,
                                                                        start_date_database=INITIAL_DATE_OF_DATASET,
                                                                        add_vix=ADD_VIX,
                                                                        calculate_ratio_close_with_indicators=CALCULATE_RATIOS)

    if price_history_df.empty:
        st.error('Could not query price history.', icon="🚨")
        sys.exit(1)
        st.stop()

    # prepare data from the model
    feature_list_nested = [predictors_dict[feature] for feature in SELECTED_FEATURES]
    FEATURE_LIST = [element for sublist in feature_list_nested for element in sublist]
    last_sequence_dict, _, _, _ = prepare_data_for_ml(price_history_df,
                                                      feature_list=FEATURE_LIST,
                                                      horizon_days_prediction=LOOKUP_STEP,
                                                      previous_days_history=PREVIOUS_DAYS_HISTORY,
                                                      scale=SCALE, lstm=LSTM)

    # calculate basic stats on the ticker
    num_days_in_df = len(price_history_df["is positive change"])
    num_positive_days_in_df = np.sum(price_history_df["is positive change"])
    if TICKER == "SPY":
        # SPY stats
        number_positive_days = 625 + num_positive_days_in_df
        number_negative_days = 532 + (num_days_in_df - num_positive_days_in_df)

    # ======================================================================================
    #                           PREDICT PRICE (PRODUCT BUSINESS LOGIC)
    # ======================================================================================

    # define model input (here I can select one specific close from the list)
    model_input = last_sequence_dict["sequence"]
    number_of_days_to_predict = model_input.shape[0]
    price_future = trained_model(model_input).numpy().tolist()
    if SCALE:
        price_future = np.squeeze(column_scaler_loaded["close"].inverse_transform(price_future), 1)
        last_sequence_dict["close"] = np.squeeze(column_scaler_loaded["close"].inverse_transform(np.expand_dims(last_sequence_dict["close"], 1)), 1)
    # apply bias
    price_future = price_future + final_bias

    if len(price_future) == 0:
        st.error('Internal error: could not predict price.', icon="🚨")
        sys.exit(1)
        st.stop()
    else:
        st.success('Prediction successful! ', icon="✅")

    # Show results in the app
    st.markdown("<h2 style='text-align: center; '>Prediction</h2>", unsafe_allow_html=True)

    is_positive_change, pct_change, category_pct_change = calc_relative_change_with_levels(last_sequence_dict["close"], price_future)
    for idx, predicted_price in enumerate(price_future):
        close_day = last_sequence_dict["date close"][idx]
        close_price = last_sequence_dict["close"][idx]
        change_pct = 100. * (predicted_price - close_price) / close_price
        direction = "POSITIVE" if change_pct > 0 else "NEGATIVE"
        st.markdown("<h4 style='text-align: center;'>" + f"On the {close_day.strftime('%d-%b-%Y')}, {TICKER} closed at {close_price:.1f}."  + " </h4>",
                    unsafe_allow_html=True)
        if TYPE_OF_PREDICTION == "price":
            if LOOKUP_STEP == 1:
                st.markdown("<h4 style='text-align: center;'>" +
                            f"The closing price at the next trading day will be: {predicted_price:.1f} ({change_pct:+.1f}%)"
                            + " </h4>", unsafe_allow_html=True)
        elif TYPE_OF_PREDICTION == "change":
            if LOOKUP_STEP == 1:
                st.markdown("<h4 style='text-align: center;'>" + f"The next trading day will be: {direction}" + " </h4>", unsafe_allow_html=True)

    # PLOT
    # requires train_test_dict = split_train_test_lstm(...)
    # test_df = train_test_dict["test_df"]
    # y_close_test = test_df["close"].values
    # if SCALE:
    #     y_close_test = np.squeeze(column_scaler_loaded["close"].inverse_transform(np.expand_dims(y_close_test, axis=0)))
    # plot_price_prediction(price_future, y_close_test, last_sequence_dict["close"])

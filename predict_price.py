# Copyright (c) 2024 Jacopo Ventura

import pickle
import os
from src.long_short_term_memory.model import *
from src.long_short_term_memory.helpers_lstm import *
from src.common.helpers_common import *
import streamlit as st
# import streamlit_analytics
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import warnings


# TODO:
# 1. stats spy since 01.01.2020
# -> num of positive days & nums of negative days to be added to the last query

# 2. query HISTORY + 2 days
# 3. current spy statistics

# 4. move to streamlit

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)
warnings.filterwarnings("ignore")

# inputs
TICKER = "SPY"   # input via streamlit
LOOKUP_STEP = 1  # input via streamlit
TYPE_OF_PREDICTION = "price"  # or change, input via streamlit

MODEL_FILENAME = "LSTM_" + TICKER + f"_lookup_{LOOKUP_STEP}"
MODEL_PATH = os.path.join("trained models", MODEL_FILENAME)

# load model parameters
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


INITIAL_DATE_OF_DATASET = datetime.datetime(2021, 1, 1).date()
INITIAL_DATE_OF_TEST = datetime.datetime(2024, 1, 1).date()




# query price data
# STEP 1: query price history and calculate TA
price_history_df, predictors_dict = query_and_calculate_ta_features(ticker=TICKER,
                                                                    start_date_database=INITIAL_DATE_OF_DATASET,
                                                                    add_vix=ADD_VIX,
                                                                    calculate_ratio_close_with_indicators=CALCULATE_RATIOS)

# STEP 2: prepare data from the model
feature_list_nested = [predictors_dict[feature] for feature in SELECTED_FEATURES]
FEATURE_LIST = [element for sublist in feature_list_nested for element in sublist]
last_sequence_dict, X, y, column_scaler = prepare_data_for_ml(price_history_df,
                                                              feature_list=FEATURE_LIST,
                                                              horizon_days_prediction=LOOKUP_STEP,
                                                              previous_days_history=PREVIOUS_DAYS_HISTORY,
                                                              scale=SCALE, lstm=LSTM)


# print behavior of the price (requires step 2)
behavior_of_the_price(price_history_df)


# Load model
final_model = tf.saved_model.load(MODEL_PATH)


# ======================================================================================
#                           MODEL USAGE (PRODUCT BUSINESS LOGIC)
# ======================================================================================


# define model input (here I can select one specific close from the list)
model_input = last_sequence_dict["sequence"]
number_of_days_to_predict = model_input.shape[0]
price_future = final_model(model_input).numpy().tolist()
if SCALE:
    price_future = np.squeeze(column_scaler["close"].inverse_transform(price_future), 1)
    last_sequence_dict["close"] = np.squeeze(column_scaler["close"].inverse_transform(np.expand_dims(last_sequence_dict["close"], 1)), 1)
# apply bias
price_future = price_future + final_bias

is_positive_change, pct_change, category_pct_change = calc_relative_change_with_levels(last_sequence_dict["close"], price_future)
for idx, predicted_price in enumerate(price_future):
    close_day = last_sequence_dict["date close"][idx]
    close_price = last_sequence_dict["close"][idx]
    change_pct = 100. * (predicted_price - close_price) / close_price
    print(f"Price {LOOKUP_STEP} days after the close {close_price:.1f} on the " + str(close_day) + f": {predicted_price:.1f} , "
                                                                                                   f"thus {change_pct:+.1f}%")



# PLOT
# requires train_test_dict = split_train_test_lstm(...)
# test_df = train_test_dict["test_df"]
# y_close_test = test_df["close"].values
# if SCALE:
#     y_close_test = np.squeeze(column_scaler["close"].inverse_transform(np.expand_dims(y_close_test, axis=0)))
# plot_price_prediction(price_future, y_close_test, last_sequence_dict["close"])

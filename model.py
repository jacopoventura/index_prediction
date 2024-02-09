import datetime
import itertools
import multiprocessing
import pandas as pd
import pickle
from src.helper_ml_train_test import create_and_test_random_forest, train_and_deploy, query_and_prepare_dataset, check_prediction_probability_binary

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)
print("Number of available cores: ", multiprocessing.cpu_count(), "\n")

# TO DO
# 0. SELECT LAST DATE CORRECTLY !!!!
# 0. Usage of prediction positive (1) & prediction negative (1) together ?
# 1. Most important hyperparameters of Random Forest + Understand how to evaluate prediction negative day
# 2. add USD/EUR data
# 3. try to normalize the data
# 4. better understand the data and try to improve with better predictors
# 5. Hyper parameterization: parameters of the model, length of training window, etc...
# 6. Try a NN:
#   6a: the input is an array with values of the predictors. Here predictors should include the tendency of the movement
#   6b: the input is an array with the last N days (represented by the predictor values as in 6a). The input is the last N days of 10a


# #######################################################################
# Choice of predictors:
# VIX: is used because low vix usually means price increase
# Daily movements with respect to previous close: relative change does not change with the time (like the price). Also, patterns (like recovery
#       from negative lows) can be recognized and used.
# Day of the year: possible info on seasonality (ex: october i the most bearish month for stocks)
# #######################################################################


# ============================================= PARAMETERS =====================================
PREDICTION_TARGET = "negative"
TRAINING_DAYS_INITIAL = 1000  # number of days for the first training
TEST_DAYS_STEP = 125  # number of days for the testing the prediction (frequency of new training).
# It indicates how often we should train again the model with most recent data
THRESHOLD_PROBABILITY_POSITIVE_CLASS = 0.6
INITIAL_DATE_OF_DATASET = datetime.datetime(2000, 1, 1).date()

# ============================================== DATASET =======================================
sp500, predictors_dict, last_closed_trading_day = query_and_prepare_dataset(ticker="^GSPC",
                                                                            prediction_target=PREDICTION_TARGET,
                                                                            start_date=INITIAL_DATE_OF_DATASET,
                                                                            previous_days_history=[5])

# behavior of the index
print("============================== behavior of the index =================================")
print("Number of trading days in the dataset: ", sp500.shape[0])
number_of_days_tested = sp500.iloc[TRAINING_DAYS_INITIAL:].shape[0]
percentage_positive_days = sp500["Target"].iloc[TRAINING_DAYS_INITIAL:].sum() / number_of_days_tested
if PREDICTION_TARGET == "negative":
    percentage_positive_days = 1 - percentage_positive_days
percentage_negative_days = 1 - percentage_positive_days
print("Number of trading days used to test the model: ", number_of_days_tested)
print(f"Percentage of positive days in the dataset: {percentage_positive_days:.2f}")
print(f"Percentage of negative days in the dataset: {percentage_negative_days:.2f}")
print(" ")

# This plot is crashing the script due to a bug with macOS
# sp500_plot = sp500.plot.line(y="Close", use_index=True)
# plt.show() # plt in place of ax
predictors_price_change = ["Close/PDclose", "Open/PDclose", "High/PDclose", "Low/PDclose", "Volume/PDvolume"]

dict_predictors = {"price": ["open", "close", "high", "low", "volume"],
                   "price change": predictors_price_change,
                   "MA": predictors_dict["ma"],
                   "price change, MA": predictors_price_change + predictors_dict["ma"],
                   "price change, rsi": predictors_price_change + predictors_dict["rsi"],
                   "price change, MA, rsi": predictors_price_change + predictors_dict["ma"] + predictors_dict["rsi"],
                   "price change, MA, rsi, vix": predictors_price_change + predictors_dict["ma"] + predictors_dict["rsi"] + ["vix", "vix/sma"],
                   "price change, MA, rsi, vix, day": predictors_price_change +
                                                      predictors_dict["ma"] +
                                                      predictors_dict["rsi"] + ["vix", "vix/sma"] + ["Day of year"],
                   "price change, MA, rsi, vix, previous days": predictors_price_change +
                                                                predictors_dict["ma"] +
                                                                predictors_dict["rsi"] + ["vix", "vix/sma"] +
                                                                predictors_dict["past features"]
                   }

estimators = [200]
sample_splits = [50]
hyperparameters = [estimators, sample_splits]
hyperparameters_combinations = list(itertools.product(*hyperparameters))

for key, predictors in dict_predictors.items():
    for combination in hyperparameters_combinations:
        n_estimators = combination[0]
        n_sample_splits = combination[1]
        print("============================== predictors: " + key + " ====================================")
        print("Estimators: ", n_estimators, "sample splits: ", n_sample_splits)
        create_and_test_random_forest(sp500, predictors,
                                      n_estimators, n_sample_splits,
                                      TRAINING_DAYS_INITIAL, TEST_DAYS_STEP, THRESHOLD_PROBABILITY_POSITIVE_CLASS)
        if key == "price":
            print("NOTE: performance of the basic model are poor because it is trained with absolute values of the index. \n"
                  "In fact, if years ago the index price was 10 and now is 100, the model hardly recognizes the patterns. \n"
                  "We need to train with price data relative to each others.\n")

# ================================== Selected model ==============================
selected_predictors = dict_predictors["price change, MA, rsi, vix"]
start_date_training = datetime.datetime(2001, 1, 1).date()
end_date_training = datetime.datetime(2023, 12, 31).date()
train_and_deploy(sp500, selected_predictors,
                 start_date_training, end_date_training,
                 200, 50, THRESHOLD_PROBABILITY_POSITIVE_CLASS)

# ================================== Deploy ==============================
# load model
filename = "RF_test.pickle"
model_loaded = pickle.load(open(filename, "rb"))
predictions = model_loaded.predict_proba([last_closed_trading_day[selected_predictors]])  # predict the probability of each possible class [0, 1]
predicted_class = check_prediction_probability_binary(predictions[:, 1], THRESHOLD_PROBABILITY_POSITIVE_CLASS)[0]
final_prediction = None
if PREDICTION_TARGET == "negative":
    if predicted_class == 1:
        final_prediction = "negative"
    else:
        final_prediction = "positive"
else:
    if predicted_class == 1:
        final_prediction = "positive"
    else:
        final_prediction = "negative"
date_prediction = last_closed_trading_day.name + datetime.timedelta(1)
print(f"The trading day after {date_prediction} will be {final_prediction}")

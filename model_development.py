import datetime
import itertools
import multiprocessing
import pandas as pd
import pickle
import time
from src.common.helpers_common import query_and_prepare_dataset
from src.random_forest.helpers_random_forest import (create_and_backtest_random_forest,
                                                     check_prediction_probability_binary,
                                                     train_and_deploy_random_forest)
from src.neural_network.helpers_neural_network import create_and_backtest_neural_network, train_and_deploy_neural_network

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)
print("Number of available cores: ", multiprocessing.cpu_count(), "\n")

# ##################################################################################################################
# NEURAL NETWORKS
# Q: How many output neurons for a binary classification? 1 or 2?
# A: There is no correct or wrong choice here. The numerical value(s) outputted by the last neuron are meaningless for classification.
# We can assume that the higher the number, the higher the probability of the corresponding class. However, we need to use an activation function.
# 1 neuron: Sigmoid activation + Binary Cross-Entropy loss
# 2 neurons: Softmax* activation + Categorical Cross-Entropy loss
# *Softmax: maps a vector into a vector of the same size, whose elements are in [0;1] and sum 1
# Source: https://towardsdatascience.com/deep-learning-which-loss-and-activation-functions-should-i-use-ac02f1c56aa8

# Q: why using batches in training ?
# A: Epoch: during the training, the model visits the entire training set.
# The gradient can be calculated by using all the training data or just a subset of it. This subset is called batch or mini-batch.
# Batch Gradient Descent (or Gradient Descent): the gradient is computed using the entire training set and the model is updated few times.
# Issues with BGD:
# 1. huge memory to compute the gradient
# 2. if non-convex function to minimize, risk to hit a local minima
# By using small batches, we reduce memory consumption since the gradient is updated with each batch. Also, we reduce the risk of local minima.
# ##################################################################################################################

# ##################################################################################################################
# KEY INFO
# 1. the threshold percentage to accept the predicted positive class does not influence the model's parameters (i.e. the training) directly.
#    It only influences the choice of the class (the predicted percentages are the same), thus precision and recall, thus the hyperparameters of
#    the model.
# 2. the higher the threshold the lower the false positive, but the higher the false negative
# 3. There is no difference in setting the positive class to be the positive day or the negative day. Predicted percentages will be switched.

# KPIs:
# minimize the FP -> high precision, high specificity (stretch: high sensitivity)
# precision = tp / (tp + fp):
# specificity (recall) = tn / (tn + fp): out of all the times the real class was negative (market down), how many times the model was correct
# sensitivity = tp / (tp + fn): out of all the times the real class was positive (market up), how many times the model was correct

# HOW TO achieve the KPIs:
# 1. increase the THRESHOLD_PROBABILITY_POSITIVE_CLASS: the higher the threshold, the higher the precision & specificity
# 2. increase sample splits
# 3. increase the number of estimators
# 4. reduce TEST_DAYS_STEP ?
# ##################################################################################################################

# ##################################################################################################################
# Choice of predictors:
# VIX: is used because low vix usually means price increase
# Daily movements with respect to previous close: relative change does not change with the time (like the price). Also, patterns (like recovery
#       from negative lows) can be recognized and used.
# Day of the year: possible info on seasonality (ex: october i the most bearish month for stocks)
# ##################################################################################################################

# ##################################################################################################################
# Hyperparameters of a random forest
# n_estimators: number of trees of the forest. The higher this number, the more accurate the model, at cost of more training time.
# max_features: number of features that are considered when splitting a node in a decision tree. The higher this number, the more accurate model,
# at the cost of overfitting.
# min_samples_split: minimum number of samples required to split a node in a decision tree. The higher this number, the more robust the model,
# at the cost of underfitting.
# min_samples_leaf: minimum number of samples required to be in a leaf node. The higher this number, the more conservative the model,
# at the cost of underfitting.
# ##################################################################################################################

# ============================================= PARAMETERS =====================================
PREDICTION_TARGET = "positive"
HORIZON_DAYS_PREDICTION = 1
TRAINING_DAYS_INITIAL = 1000  # number of days for the first training
TEST_DAYS_STEP = 100  # number of days for the testing the prediction (frequency of new training).
# It indicates how often we should train again the model with most recent data
THRESHOLD_PROBABILITY_POSITIVE_CLASS = 0.6
NUMBER_OF_DAYS_PREVIOUS_DATA = 15
INITIAL_DATE_OF_DATASET = datetime.datetime(2000, 1, 1).date()

# ============================================== DATASET =======================================
sp500, predictors_dict, last_closed_trading_day = query_and_prepare_dataset(ticker="^GSPC",
                                                                            prediction_target=PREDICTION_TARGET,
                                                                            horizon_days_prediction=HORIZON_DAYS_PREDICTION,
                                                                            start_date=INITIAL_DATE_OF_DATASET,
                                                                            previous_days_history=[NUMBER_OF_DAYS_PREVIOUS_DATA])

#dataframe_size_mb = sp500.memory_usage(index=True).sum() / 1000000.
#print(dataframe_size_mb)
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
predictors_price_change = ["Close/PDclose"]  # , "Open/PDclose", "High/PDclose", "Low/PDclose", "Volume/PDvolume"]

dict_predictors = {"price": ["open", "close", "high", "low", "volume"],
                   "close": ["close"],
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
                                                                predictors_dict["rsi"] + ["vix", "vix/sma"] + # + ["macd", "atr"] +
                                                                predictors_dict["past features"]
                   }


# ============================================== Neural Network =================================
predictors = dict_predictors["price change, MA, rsi, vix, previous days"]

# build a binary classifier NN pytorch
# https://machinelearningmastery.com/building-a-binary-classification-model-in-pytorch/

parameters_neural_network = {"epochs": 50,
                             "batch_size": 32,
                             "learning_rate": 1e-4,
                             "weight_decay": 1e-5
                             }

#start = time.time()
#create_and_backtest_neural_network(sp500, predictors, parameters_neural_network,
#                                   TRAINING_DAYS_INITIAL, TEST_DAYS_STEP, THRESHOLD_PROBABILITY_POSITIVE_CLASS)
#duration = time.time() - start
#print(duration)

print(" ########################### NEURAL NETWORK ###################################")
for key, predictors in dict_predictors.items():
    print("============================== predictors: " + key + " ====================================")
    # print("Estimators:", n_estimators, "  sample splits:", n_sample_splits, "  min sample leafs:", n_min_samples_leaf)
    # create_and_backtest_neural_network(sp500, predictors, parameters_neural_network,
    #                                   TRAINING_DAYS_INITIAL, TEST_DAYS_STEP, THRESHOLD_PROBABILITY_POSITIVE_CLASS)


print("")

# ============================================== Random Forest =================================
print(" ########################### RANDOM FOREST ###################################")

# sp500["vix"] = sp500["vix"] / 80
estimators = [200]
sample_splits = [100]
min_samples_leaf = [1]
hyperparameters = [estimators, sample_splits, min_samples_leaf]
hyperparameters_combinations = list(itertools.product(*hyperparameters))

for key, predictors in dict_predictors.items():
    for combination in hyperparameters_combinations:
        n_estimators = combination[0]
        n_sample_splits = combination[1]
        n_min_samples_leaf = combination[2]
        parameters_random_forest = {"n_estimators": n_estimators,
                                    "n_samples_split": n_sample_splits,
                                    "n_min_samples_leaf": n_min_samples_leaf}
        print("============================== predictors: " + key + " ====================================")
        print("Estimators:", n_estimators, "  sample splits:", n_sample_splits, "  min sample leafs:", n_min_samples_leaf)
        create_and_backtest_random_forest(sp500, predictors, parameters_random_forest,
                                          TRAINING_DAYS_INITIAL, TEST_DAYS_STEP, THRESHOLD_PROBABILITY_POSITIVE_CLASS)
        if key == "price":
            print("NOTE: performance of the basic model are poor because it is trained with absolute values of the index. \n"
                  "In fact, if years ago the index price was 10 and now is 100, the model hardly recognizes the patterns. \n"
                  "We need to train with price data relative to each others.\n")

# ================================== Selected model ==============================
filename = f"positive_or_negative_{HORIZON_DAYS_PREDICTION}days_RF.pickle"
selected_predictors = dict_predictors["price change, MA, rsi, vix, previous days"]
start_date_training = datetime.datetime(2001, 1, 1).date()
end_date_training = datetime.datetime(2023, 12, 31).date()
parameters_random_forest = {"n_estimators": estimators[0],
                            "n_samples_split": sample_splits[0],
                            "n_min_samples_leaf": min_samples_leaf[0]}
train_and_deploy_random_forest(sp500, selected_predictors, parameters_random_forest,
                               start_date_training, end_date_training, filename,
                               THRESHOLD_PROBABILITY_POSITIVE_CLASS)

train_and_deploy_neural_network(sp500, selected_predictors, parameters_neural_network,
                                start_date_training, end_date_training, "test",
                                THRESHOLD_PROBABILITY_POSITIVE_CLASS)

exit()

# ================================== Deploy ==============================
# load model
model_loaded = pickle.load(open(filename, "rb"))
predictions = model_loaded.predict_proba([last_closed_trading_day[selected_predictors]])  # predict the probability of each possible class [0, 1]
print(predictions)
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

print(f"In {HORIZON_DAYS_PREDICTION} trading days after {last_closed_trading_day.name}, the price change will be"
      f" {final_prediction}")

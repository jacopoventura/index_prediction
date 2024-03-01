import datetime
import itertools
import multiprocessing as mp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle
import time
import ray
import warnings
from src.pattern_matching.pattern_matching import PatternMatching
from src.common.helpers_common import query_and_prepare_dataset, split_dataset_with_date
from src.random_forest.helpers_random_forest import (create_and_backtest_random_forest,
                                                     check_prediction_probability_binary,
                                                     train_and_deploy_random_forest)
from src.neural_network.helpers_neural_network import create_and_backtest_neural_network, train_and_deploy_neural_network



pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)
warnings.filterwarnings("ignore")
# print("Number of available cores: ", mp.cpu_count(), "\n")

# TO DO:
# 2. Best NN vs best RF
# goal: 55%+ accuracy + 50%+ precision


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

# Q: why many trained model have 0 scores (no FP / FN)?
# A: the performance of a trained NN depend on the choice of hyperparameters: learning rate, number of layers / neurons, optimizer, etc.
# We noticed that, for example, setting a high POSITIVE_THRESHOLD, we often get a trained model with 0 precision,
# because the probability threshold is so high, that the NN never outputs the positive class (in complex scenarios, the probability is never high)

# Q: why scaling is important?
# A: Scaling the data balances the impact of all variables on the distance calculation and can help to improve the performance.
# Several ML techniques like neural networks require the input data to be normalized to work well.
# If input data are normalized, the weights are of the same order of magnitude. In this way, the optimization solver is more stable
# and the input predictors have the same impact on the prediction.
# ##################################################################################################################

# ##################################################################################################################
# KEY INFO
# 1. the threshold percentage to accept the predicted positive class does not influence the model's parameters (i.e. the training) directly.
#    It only influences the choice of the class (the predicted percentages are the same), thus precision and recall, thus the hyperparameters of
#    the model.
# 2. the higher the threshold the lower the false positive, but the higher the false negative
# 3. There is no difference in setting the positive class to be the positive day or the negative day. Predicted percentages will be switched.

# KPIs:
# precision = tp / (tp + fp): success probability of making a correct positive class classification
# specificity (recall) = tn / (tn + fp): out of all the times the real class was negative (market down), how many times the model was correct
# sensitivity = tp / (tp + fn): out of all the times the real class was positive (market up), how many times the model was correct
# accuracy = (tp + tn) / population: how comfortable the model is with detecting both positive and negative classes.
# Sources:
# https://medium.com/@yashwant140393/the-3-pillars-of-binary-classification-accuracy-precision-recall-d2da3d09f664
# https://medium.com/@satyarepala/understanding-the-confusion-matrix-a-practical-guide-to-validation-metrics-for-binary-classifiers-8062a59613e6
#
# KPI requirements:
# accuracy > 0.5: 0.5 is random (coin toss)
# precision > 0.46: 0.46 is the historical probability of a positive day
# specificity > 0.54: 0.54 is the historical probability of a positive day


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

if __name__ == '__main__':
    # ============================================= PARAMETERS =====================================
    PREDICTION_TARGET = "positive"
    HORIZON_DAYS_PREDICTION = 1
    TRAINING_DAYS_INITIAL = 1500  # number of days for the first training
    TEST_DAYS_STEP = 125  # number of days for the testing the prediction (frequency of new training).
    # It indicates how often we should train again the model with most recent data
    THRESHOLD_PROBABILITY_POSITIVE_CLASS = 0.535  # .55
    NUMBER_OF_DAYS_PREVIOUS_DATA = 15
    INITIAL_DATE_OF_DATASET = datetime.datetime(2000, 1, 1).date()

    # ============================================== DATASET =======================================
    sp500, predictors_dict, last_closed_trading_day = query_and_prepare_dataset(ticker="^GSPC",
                                                                                prediction_target=PREDICTION_TARGET,
                                                                                horizon_days_prediction=HORIZON_DAYS_PREDICTION,
                                                                                start_date=INITIAL_DATE_OF_DATASET,
                                                                                previous_days_history=[NUMBER_OF_DAYS_PREVIOUS_DATA])
    sp500["vix"] = sp500["vix"] / 80
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

    # ############################### PATTERN #####################
    #plt.figure(figsize=(16,5))
    #plt.style.use("seaborn-v0_8-whitegrid")
    #plt.plot(sp500.iloc[2000:2100]["Target"], label="index", c="#4ac2fb")
    #plt.plot(sp500.iloc[2001:2010]["Target"], lw=3, label="Pattern (9d)", c="#ff4e97")
    #plt.legend(frameon=True, fancybox=True, framealpha=.9, loc=1)
    #plt.title("Pattern in testing set", fontsize=15)
    #plt.ylabel("return")
    # plt.show()



    train_dataset, test_dataset = split_dataset_with_date(sp500, INITIAL_DATE_OF_DATASET, datetime.datetime(2023, 9, 30).date())
    # pattern_matching_algo = PatternMatching(train_dataset["Target"].to_list())

    # Standard python
    #pattern_lengths = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    #start = time.time()
    #df = pattern_matching_algo.pattern_matching_advanced(pattern_lengths)
    #end_python = time.time() - start
    #print("python: ", end_python)


    # pattern_matching_original(binary_train, binary_test)
    pattern_length_list = [[5, 6], [7, 8], [9, 10], [11, 12], [13], [14], [15], [16]]


    ray.init()
    pattern_matching_algo_ray = PatternMatching.remote(train_dataset["Target"].to_list())

    start = time.time()
    object_references = [
        pattern_matching_algo_ray.pattern_matching_advanced.remote(item) for item in pattern_length_list
    ]
    data = ray.get(object_references)
    duration_ray = time.time() - start
    print("ray: ", duration_ray)

    #start = time.time()
    #with mp.Pool() as pool:
    #    dfu = pool.map(pattern_matching_algo.pattern_matching_advanced, pattern_length_list)
    ##futures = [pattern_matching_advanced.remote(train_dataset["Target"].to_list(), test_dataset["Target"].to_list(), i) for i in
    ## pattern_length_list]
    ##r = ray.get(futures)
    #duration_multiprocess = time.time() - start
    #print("multiprocess: ", duration_multiprocess)


    #pattern_matching_advanced(train_dataset["Target"].to_list(), test_dataset["Target"].to_list())


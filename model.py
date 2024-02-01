import yfinance as yf
import multiprocessing
import pandas as pd
import datetime
from src.finance_algo import calculate_rsi
from src.helper_ml_train_test import create_and_test_random_forest, train_and_deploy, add_previous_behavior
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)
print("Number of available cores: ", multiprocessing.cpu_count(), "\n")

# TO DO
# 1. check score. Provide one score every new train. Save in list, then final results: sliding or accumulating training ?
# 2. Clarify prediction negative day
# 3. How do I use this model daily ? Train and code daily run of the algo. Check when to run: after close?
# 4. Hyper parameterization: parameters of the model, length of training window, etc...
# 5. consider daily change prediction (positive, -0.5pct, -1pct, -2pct, -3pct, more than -3pct)
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
TRAINING_DAYS_INITIAL = 1000  # number of days for the first training
TEST_DAYS_STEP = 125  # number of days for the testing the prediction. Also frequency of new training.
# It indicates how often we should train again the model with most recent data
THRESHOLD_PROBABILITY_POSITIVE = 0.55
HORIZON_MA_VIX = 5
INITIAL_DATE_OF_DATASET = datetime.datetime(2000, 1, 1).date()


# ============================================== DATASET =======================================

# Query the historical data of the index from yahoo finance
sp500 = yf.Ticker("^GSPC")
sp500 = sp500.history(period="max")

# Extract date from index and set date as index (VIX has another hour, thus dataframes cannot be directly merged)
sp500["Trading date"] = [d.date() for d in sp500.index.to_list()]
sp500.set_index("Trading date", inplace=True)

# Query the historical data of VIX
vix = yf.Ticker("^VIX")
vix = vix.history(period="max")
# Extract date from index and set date as index
vix["Trading date"] = [d.date() for d in vix.index.to_list()]
vix.set_index("Trading date", inplace=True)
vix.rename(columns={'Close': 'VIX'}, inplace=True)

# we consider only the most recent data. The market could have behaved differently many decades ago
sp500 = sp500.loc[vix.index[0]:].copy()

# Remove useless columns (relevant for single stocks only)
del sp500["Dividends"]
del sp500["Stock Splits"]

# Concatenate VIX
sp500 = pd.concat([sp500, vix[["VIX"]]], axis=1)

# Calculate RSI
periods_rsi = [5, 14]
predictors_rsi = []
for period in periods_rsi:
    column_name_rsi = f"RSI{period}"
    sp500[column_name_rsi] = calculate_rsi(sp500["Close"], period=period)
    predictors_rsi += [column_name_rsi]


# Calculate daily price movement with respect to previous day close
sp500_previous_day_close = sp500["Close"].shift(1)
sp500["Close/PDclose"] = sp500["Close"] / sp500_previous_day_close
sp500["Open/PDclose"] = sp500["Open"] / sp500_previous_day_close
sp500["High/PDclose"] = sp500["High"] / sp500_previous_day_close
sp500["Low/PDclose"] = sp500["Low"] / sp500_previous_day_close

# Calculate VIX change with respect to previous day
sp500["VIX/PDVIX"] = sp500["VIX"] / sp500["VIX"].shift(1)

# Calculate Volume change with respect to previous day
sp500["Volume/PDvolume"] = sp500["Volume"] / sp500["Volume"].shift(1)

# Calculate the day of the year
dates = sp500.index.to_list()
sp500["Day of year"] = [d.timetuple().tm_yday for d in dates]

# # Utest
# num_rows = sp500.shape[0]
# sum_errors = 0
# for i in range(1,num_rows-1):
#     sum_errors += sp500["Close/prev day close"].iloc[i] - (sp500["Close"].iloc[i] / sp500["Close"].iloc[i-1])
#     sum_errors += sp500["Open/prev day close"].iloc[i] - (sp500["Open"].iloc[i] / sp500["Close"].iloc[i - 1])
#     sum_errors += sp500["High/prev day close"].iloc[i] - (sp500["High"].iloc[i] / sp500["Close"].iloc[i - 1])
#     sum_errors += sp500["Low/prev day close"].iloc[i] - (sp500["Low"].iloc[i] / sp500["Close"].iloc[i - 1])
# print(sum_errors)


# The ML model shall predict if the price of tomorrow will be higher or lower
# Create a column with the closing price of the day after (which is the movement we want to predict)
# Target: 0 if negative day, 1 if positive day
sp500["Tomorrow"] = sp500["Close"].shift(-1)
sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)

# Associate
features_to_associate = ["Close/PDclose", "Open/PDclose"]
sp500, past_features_associated_today_list = add_previous_behavior(sp500, [1, 2], features_to_associate)

# select specific interval
sp500 = sp500.loc[INITIAL_DATE_OF_DATASET:].copy()

# remove rows containing at least 1 NaN
sp500 = sp500.dropna()
# NOTE: we are dealing with time-series data. Cross validation is not useful

# behavior of the index
print("============================== behavior of the index =================================")
print("Number of trading days in the dataset: ", sp500.shape[0])
number_of_days_tested = sp500.iloc[TRAINING_DAYS_INITIAL:].shape[0]
percentage_positive_days = sp500["Target"].iloc[TRAINING_DAYS_INITIAL:].value_counts() / number_of_days_tested
print("Number of trading days used to test the model: ", number_of_days_tested)
print("Percentage of positive days in the dataset: ", percentage_positive_days[1])
print(" ")

# This plot is crashing the script due to a bug with macOS
# sp500_plot = sp500.plot.line(y="Close", use_index=True)
# plt.show() # plt in place of ax


# ============================ BASELINE MODEL WITH INDEX VALUES =========================
# We first train a basic model that will be used as benchmark.
# This benchmark model is trained with the original data of the dataset: volume, open, high, low, close prices.
# define data to be used by the ML model to predict the output
predictors_basic = ["Close", "Volume", "Open", "High", "Low"]

# ML model: random forest classifier. Reasons:
# 1. resistant to overfit because of the numerous random trees
# 2. run quickly
# 3. handles non-linear relationships
print("============================== basic model based on price ====================================")
create_and_test_random_forest(sp500, predictors_basic,
                              200, 50,
                              TRAINING_DAYS_INITIAL, TEST_DAYS_STEP, THRESHOLD_PROBABILITY_POSITIVE)
print("NOTE: performance of the basic model are poor because it is trained with absolute values of the index. \n"
      "In fact, if years ago the index price was 10 and now is 100, the model hardly recognizes the patterns. \n"
      "We need to train with price data relative to each others.\n")


# ========================= ADVANCED MODEL 1: PREDICTION WITH RELATIVE PRICE CHANGE =======================
# We first train a basic model that is used as benchmark.
# This benchmark model is trained with the original data of the dataset: volume, open, high, low, close prices.
# The model does not predict well because we used price data directly. If 10 years ago the price was 10 and now is 1000,
# then the patterns are hardly recognized by the model.
# We now train a model with prices relative to the previous day.
predictors_price_movement = ["Close/PDclose",
                             "Open/PDclose",
                             "High/PDclose",
                             "Low/PDclose",
                             "Volume/PDvolume"
                             ]

print("============================== model based on relative price movement ==================================")
create_and_test_random_forest(sp500, predictors_price_movement,
                              200, 50,
                              TRAINING_DAYS_INITIAL, TEST_DAYS_STEP, THRESHOLD_PROBABILITY_POSITIVE)


# ============================ ADVANCED MODEL 2: PREDICTION WITH MA ONLY =========================
# Now we build a model that uses rations with the Moving Averages (MA) as predictors. Prices will not be used in this model.
horizons_days_moving_average = [5, 50, 100]
predictors_ma = []

# Loop over the MA horizons and calculate the ratios
for horizon in horizons_days_moving_average:

    # calculate the moving average
    column_name_ratio_with_ma = f"Close_ration_with_{horizon}MA"

    # Pandas' rolling function implements the sliding window operation.
    # Let N be the size of the window, then the window at the ith row will be [i+1-N:i+1]
    # This means that the sliding windows is built from the current index i plus the previous (N-1) elements (rows)
    # This makes sense because the moving average of price must consider the closing of the current day
    # NOTE: if we want to calculate the MA without the current day close, we need to do dataframe.shift(1).rolling()
    moving_averages = sp500.rolling(horizon).mean()

    # add the column day close / MA close to the dataset
    # we calculate the ratio close / MA close. Notably, the ma close is calculated using the current daily close, i.e. the enumerator
    sp500[column_name_ratio_with_ma] = sp500["Close"] / moving_averages["Close"]

    # calculate the trend: sum of positive days withing the selected horizon
    column_name_trend = f"Trend_{horizon}"
    # the shift(1) shifts the row by 1. This allows to calculate the rolling excluding the current day (row)
    sp500[column_name_trend] = sp500.shift(1).rolling(horizon).sum()["Target"]

    predictors_ma += [column_name_ratio_with_ma, column_name_trend]

moving_average_vix = sp500["VIX"].rolling(HORIZON_MA_VIX).mean()
sp500["VIX MA"] = sp500["VIX"] / moving_average_vix


# remove rows containing at least 1 NaN
sp500 = sp500.dropna()

print("============================== model based on MA =================================")
create_and_test_random_forest(sp500, predictors_ma,
                              200, 50,
                              TRAINING_DAYS_INITIAL, TEST_DAYS_STEP, THRESHOLD_PROBABILITY_POSITIVE)


# ======================= ADVANCED MODEL 3: PREDICTION WITH MA AND PRICE CHANGE =========================
# Now we build a model that uses rations with the Moving Averages (MA) and relative price change as predictors.
# We combine the first two advanced models.
print("============================== model based on MA and price movement =================================")
create_and_test_random_forest(sp500, predictors_ma + predictors_price_movement,
                              200, 50,
                              TRAINING_DAYS_INITIAL, TEST_DAYS_STEP, THRESHOLD_PROBABILITY_POSITIVE)


# ======================= ADVANCED MODEL 4: PREDICTION WITH MA, PRICE CHANGE AND VIX =========================
# Now we build a model that uses rations with the Moving Averages (MA), relative price change and VIX as predictors.
# We combine the first three advanced models.
print("============================== model based on MA, price movement and VIX =================================")
create_and_test_random_forest(sp500, predictors_ma + predictors_price_movement + ["VIX"],
                              200, 50,
                              TRAINING_DAYS_INITIAL, TEST_DAYS_STEP, THRESHOLD_PROBABILITY_POSITIVE)
# Here VIX MA does not help


# ======================= ADVANCED MODEL 5: PREDICTION WITH MA, PRICE CHANGE, VIX AND DAY OF YEAR =========================
# Now we build a model that uses rations with the Moving Averages (MA), relative price change and VIX as predictors.
print("============================== model based on MA, price movement, VIX and day of year =================================")
create_and_test_random_forest(sp500, predictors_ma + predictors_price_movement + ["VIX", "VIX MA"] + ["Day of year"],
                              200, 50,
                              TRAINING_DAYS_INITIAL, TEST_DAYS_STEP, THRESHOLD_PROBABILITY_POSITIVE)


# ======================= ADVANCED MODEL 6: PREDICTION WITH MA, PRICE CHANGE, RSI AND VIX =========================
# Now we build a model that uses rations with the MAs, relative price change, RSIs and VIX as predictors.
print("============================== model based on MA, price movement, VIX and RSI =================================")
create_and_test_random_forest(sp500, predictors_ma + predictors_price_movement + ["VIX"] + predictors_rsi,
                              200, 50,
                              TRAINING_DAYS_INITIAL, TEST_DAYS_STEP, THRESHOLD_PROBABILITY_POSITIVE)


# ======================= ADVANCED MODEL 7: PREDICTION WITH MA, PRICE CHANGE, RSI, VIX AND PAST PATTERN  =========================
# Now we build a model that uses rations with the MAs, relative price change, RSIs and VIX as predictors.
print("============================== model based on MA, price movement, VIX, RSI and past movements =================================")
create_and_test_random_forest(sp500, predictors_ma + predictors_price_movement + ["VIX"] + predictors_rsi + past_features_associated_today_list,
                              200, 50,
                              TRAINING_DAYS_INITIAL, TEST_DAYS_STEP, THRESHOLD_PROBABILITY_POSITIVE)

start_date_training = datetime.datetime(2001, 1, 1).date()
end_date_training = datetime.datetime(2023, 12, 31).date()
train_and_deploy(sp500, predictors_ma + predictors_price_movement + ["VIX"] + predictors_rsi,
                 start_date_training, end_date_training,
                 500, 50, THRESHOLD_PROBABILITY_POSITIVE)

# STEPS to deploy

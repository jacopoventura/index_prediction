import yfinance as yf
from helper_train_test import create_and_test_random_forest
import pandas as pd
pd.set_option('display.max_columns', None)

# https://www.youtube.com/watch?v=1O_BenficgE
# TO DO
# 2. Train new model with MA of VIX
# 3. add RSI https://medium.com/@farrago_course0f/using-python-and-rsi-to-generate-trading-signals-a56a684fb1
# 4. Train new model with RSI
# 5. How do I use this model daily ?
# 6. Hyper parameterization: parameters of the model, length of training window, etc...
# 7. Clarify prediction negative day
# 8. Add other predictors
# 9. consider daily change prediction (positive, -0.5pct, -1pct, -2pct, -3pct, more than -3pct)
# 10. Try a NN:
#   10a: the input is an array with values of the predictors. Here predictors should include the tendency of the movement
#   10b: the input is an array with the last N days (represented by the predictor values as in 10a). The input is the last N days of 10a


# ============================================= PARAMETERS =====================================
TRAINING_DAYS_INITIAL = 2500  # number of days for the first training
TEST_DAYS_STEP = 250  # number of days for the testing the prediction.
# It indicates how often we should train again the model with most recent data
THRESHOLD_PROBABILITY_POSITIVE = 0.6


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
# sp500 = sp500.loc["1995-01-01":].copy()
sp500 = sp500.loc[vix.index[0]:].copy()

# Remove useless columns (relevant for single stocks only)
del sp500["Dividends"]
del sp500["Stock Splits"]

# Concatenate VIX
sp500 = pd.concat([sp500, vix[["VIX"]]], axis=1)

# Calculate daily price movement with respect to previous day close
sp500_previous_day_close = sp500["Close"].shift(1)
sp500["Close/prev day close"] = sp500["Close"] / sp500_previous_day_close
sp500["Open/prev day close"] = sp500["Open"] / sp500_previous_day_close
sp500["High/prev day close"] = sp500["High"] / sp500_previous_day_close
sp500["Low/prev day close"] = sp500["Low"] / sp500_previous_day_close

# Calculate VIX change with respect to previous day
sp500["VIX/prev day VIX"] = sp500["VIX"] / sp500["VIX"].shift(1)

# Calculate Volume change with respect to previous day
sp500["Volume/prev day volume"] = sp500["Volume"] / sp500["Volume"].shift(1)

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
predictors_price_movement = ["Close/prev day close",
                             "Open/prev day close",
                             "High/prev day close",
                             "Low/prev day close",
                             "Volume/prev day volume"]

print("============================== model based on relative price movement ==================================")
create_and_test_random_forest(sp500, predictors_price_movement,
                              200, 50,
                              TRAINING_DAYS_INITIAL, TEST_DAYS_STEP, THRESHOLD_PROBABILITY_POSITIVE)


# ============================ ADVANCED MODEL 2: PREDICTION WITH MA ONLY =========================
# Now we build a model that uses rations with the Moving Averages (MA) as predictors. Prices will not be used in this model.
horizons_days_moving_average = [2, 5, 22, 50, 100, 200, 250]
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


# behavior of the model
print("============================== model based on MA, price movement and VIX =================================")
create_and_test_random_forest(sp500, predictors_ma + predictors_price_movement + ["VIX"],
                              200, 50,
                              TRAINING_DAYS_INITIAL, TEST_DAYS_STEP, THRESHOLD_PROBABILITY_POSITIVE)

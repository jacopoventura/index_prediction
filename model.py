import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from helper_train_test import backtest
import pandas as pd

# https://www.youtube.com/watch?v=1O_BenficgE
# TO DO:
# 0. Add to my GitHub repo
# 1. Train new model with MA and VIX
# 2. add RSI https://medium.com/@farrago_course0f/using-python-and-rsi-to-generate-trading-signals-a56a684fb1
# 3. Train new model with RSI
# 4. Clarify prediction negative day
# 5. How do I use this model daily ?
# 6. Try a NN


# ============================================= PARAMETERS =====================================
TRAINING_DAYS_INITIAL = 2500  # number of days for the first training
TEST_DAYS_STEP = 250  # number of days for the test
THRESHOLD_PROBABILITY_POSITIVE = 0.6


# ============================================== DATASET =======================================

# Query the historical data of the index from yahoo finance
sp500 = yf.Ticker("^GSPC")
sp500 = sp500.history(period="max")

# Query the historical data of VIX
vix = yf.Ticker("^VIX")
vix = vix.history(period="max")
vix_simple = vix[["Close"]].copy()  # NOTE: to extract a sub-dataframe, we need a LIST of column names
vix_simple.rename(columns={'Close': 'VIX'}, inplace=True)

# we consider only the most recent data. The market could have behaved differently many decades ago
# sp500 = sp500.loc["1995-01-01":].copy()
sp500 = sp500.loc[vix_simple.index[0]:].copy()

# Remove useless columns (relevant for single stocks only)
del sp500["Dividends"]
del sp500["Stock Splits"]

# The ML model shall predict if the price of tomorrow will be higher or lower
# Create a column with the closing price of the day after (which is the movement we want to predict)
# Target: 0 if negative day, 1 if positive day
sp500["Tomorrow"] = sp500["Close"].shift(-1)
sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)
# NOTE: we are dealing with time-series data. Cross validation is not useful

# Concatenate index with VIX
sp500_vix = pd.concat([sp500, vix_simple], axis=1)

# behavior of the index
print("=========================== behavior of the index =================================")
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
model_basic = RandomForestClassifier(n_estimators=100,  # number of trees: the higher, the better the accuracy
                                     min_samples_split=100,  # the higher, the less accurate, but the less overfits
                                     random_state=1)  # if 1, same initialization


# Train and backtest (train inside backtest)
predictions = backtest(sp500, model_basic, predictors_basic,
                       days_initial_train=TRAINING_DAYS_INITIAL,
                       days_test=TEST_DAYS_STEP,
                       threshold_probability_positive=THRESHOLD_PROBABILITY_POSITIVE)

model_basic_precision_score = precision_score(predictions["Target"], predictions["Predictions"])


# behavior of the basic model
print("=========================== behavior of the basic model =================================")
print("Precision to predict a positive day (?): ", model_basic_precision_score)
print(" ")


# ============================ ADVANCED MODEL 1: PREDICTION WITH MA ONLY =========================

# We first train a basic model that is used as benchmark.
# This benchmark model is trained with the original data of the dataset: volume, open, high, low, close prices.
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

model_ma = RandomForestClassifier(n_estimators=200,  # number of trees: the higher, the better the accuracy
                                  min_samples_split=50,  # the higher, the less accurate, but the less overfits
                                  random_state=1)  # if 1, same initialization

# Train and backtest (train inside backtest)
predictions = backtest(sp500, model_ma, predictors_ma,
                       days_initial_train=TRAINING_DAYS_INITIAL,
                       days_test=TEST_DAYS_STEP,
                       threshold_probability_positive=THRESHOLD_PROBABILITY_POSITIVE)

model_ma_precision_score = precision_score(predictions["Target"], predictions["Predictions"])

# behavior of the model based on MA inputs
print("=========================== behavior of the model MA based =================================")
print("Precision to predict a positive day (?): ", model_ma_precision_score)
print(" ")

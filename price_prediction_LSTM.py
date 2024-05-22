import os

from src.long_short_term_memory.model import *
from src.long_short_term_memory.helpers_lstm import *
from src.common.helpers_common import *
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import warnings

# set seed, so we can get the same results after rerunning several times
np.random.seed(314)
random.seed(314)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)
warnings.filterwarnings("ignore")


# =================== MAIN ===============
ticker = "SPY"
USE_MODEL_ONLY = True
PREVIOUS_DAYS_HISTORY = 30  # best results with 25-30
LOOKUP_STEP = 1
MODEL_FILENAME = "LSTM_" + ticker + f"_horizon_{LOOKUP_STEP}_history_{PREVIOUS_DAYS_HISTORY}"
MODEL_PATH = os.path.join("trained models", MODEL_FILENAME)

# Parameters
USE_BIAS = True
LSTM = True
SCALE = True
SHUFFLE = True
SPLIT_BY_DATE = False
CALCULATE_RATIOS = True
TEST_SIZE = 0.1   # best results with 0.1, 0.15, 0.2

INITIAL_DATE_OF_DATASET = datetime.datetime(2021, 1, 1).date()
INITIAL_DATE_OF_TEST = datetime.datetime(2024, 1, 1).date()
SELECTED_FEATURES = ['hloc', 'volume', "atr"]  # , 'atr', 'macd', 'aroon', 'adx', 'bollinger',
# 'rsi',
# 'ma',
# 'day', 'price/previous day', 'vix']
# possibilities: ['hloc', 'volume', 'atr', 'macd', 'aroon', 'adx', 'bollinger', 'rsi', 'ma', 'day', 'price/previous day', 'vix']

ADD_VIX = False
if 'vix' in SELECTED_FEATURES:
    ADD_VIX = True

# model parameters
N_LAYERS = 2  # the higher, the worst the results
UNITS = 256  # best results with 256
DROPOUT = 0.4  # best results with 0.4
BIDIRECTIONAL = True
# training parameters
LOSS = "mae"
# LOSS = "huber_loss"
OPTIMIZER = "adam"
BATCH_SIZE = 64
EPOCHS = 5


# STEP 1: query price history and calculate TA
price_history_df, predictors_dict = query_and_calculate_ta_features(ticker=ticker,
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

# Train or use the model as product
if not USE_MODEL_ONLY:

    # STEP 3: create train and test dataset
    train_test_dict = split_train_test_lstm(X, y, price_history_df, FEATURE_LIST,
                                            INITIAL_DATE_OF_TEST, TEST_SIZE, SPLIT_BY_DATE, SHUFFLE)

    # STEP 4: construct the model
    model = create_lstm_model_regression(PREVIOUS_DAYS_HISTORY, len(FEATURE_LIST),
                                         number_layers=N_LAYERS, number_neurons=UNITS,
                                         dropout=DROPOUT, bidirectional=BIDIRECTIONAL,
                                         loss=LOSS, optimizer=OPTIMIZER)

    # STEP 5: train the model
    # some tensorflow callbacks
    checkpointer = ModelCheckpoint(os.path.join("results", MODEL_FILENAME + ".h5"), save_weights_only=True, save_best_only=True, verbose=1)
    tensorboard = TensorBoard(log_dir=os.path.join("logs", MODEL_FILENAME))
    # train the model and save the weights whenever we see a new optimal model using ModelCheckpoint
    X_train = train_test_dict["X_train"]
    y_train = train_test_dict["y_train"]
    X_test = train_test_dict["X_test"]
    y_test = train_test_dict["y_test"]
    history = model.fit(X_train, y_train,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_data=(X_test, y_test),
                        callbacks=[checkpointer, tensorboard],
                        verbose=1)

    bias = calc_bias(model, X_train, y_train, column_scaler, USE_BIAS, SCALE)

    # STEP 6: save the trained model
    model.save(MODEL_PATH)
    f = open(MODEL_PATH + "/bias.dat", "w+")
    f.write(str(bias))
    f.close()
    f = open(MODEL_PATH + "/scale.dat", "w+")
    f.write(str(SCALE))
    f.close()

    # STEP 7: test the trained model
    # get the final dataframe for the testing set
    y_predicted = model.predict(X_test)
    test_df = train_test_dict["test_df"]
    y_close_test = test_df["close"].values
    if SCALE:
        y_test = np.squeeze(column_scaler["close"].inverse_transform(np.expand_dims(y_test, axis=0)))
        y_predicted = np.squeeze(column_scaler["close"].inverse_transform(y_predicted))
        y_close_test = np.squeeze(column_scaler["close"].inverse_transform(np.expand_dims(y_close_test, axis=0)))
    # apply bias
    y_predicted = y_predicted + bias

    # Evaluate the scores in the test dataset
    print("\n\n============================ SCORES ==================================")
    print("Classification:")
    number_test_cases = len(y_predicted)

    # classification scores
    is_positive_change_prediction, pct_change_prediction, category_pct_change_prediction = calc_relative_change_with_levels(y_close_test,
                                                                                                                            y_predicted)
    price_difference_absolute = [np.abs(y_test[i] - y_predicted[i]) for i in range(number_test_cases)]
    pct_change_difference = [np.abs(pct_change_prediction[i] - test_df["change %"].values[i]) for i in range(number_test_cases)]
    score_dict = compute_precision_recall_specificity(test_df["is positive change"], is_positive_change_prediction)
    print_classification_scores(score_dict)
    calc_performance_category(train_test_dict["test_df"]["category % change"], category_pct_change_prediction)

    # regression scores
    mean_error_price = np.mean(price_difference_absolute)
    std_mean_error_price = np.std(price_difference_absolute)
    mean_error_pct_change = np.mean(pct_change_difference)
    std_error_pct_change = np.std(pct_change_difference)

    print(f"Mean error in price prediction: {mean_error_price} +/- {std_mean_error_price}")
    print(f"Mean error in % change prediction: {mean_error_pct_change} +/- {std_error_pct_change}")

    # plot results
    # plot_test_price_prediction(y_predicted, y_test)


# ======================================================================================
#                           MODEL USAGE (PRODUCT BUSINESS LOGIC)
# ======================================================================================

# Load model
final_model = tf.saved_model.load(MODEL_PATH)
f = open(MODEL_PATH + "/bias.dat", "r")
final_bias = float(f.read())
f.close()
f = open(MODEL_PATH + "/scale.dat", "r")
SCALE = bool(f.read())
f.close()

print("\n=========================== PRICE PREDICTION ==================================")
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

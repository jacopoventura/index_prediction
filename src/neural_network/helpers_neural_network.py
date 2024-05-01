import copy
import datetime
import pickle
import torch
import pandas as pd
from torch.utils.data import DataLoader
from src.common.helpers_common import (check_prediction_probability_binary,
                                       split_dataset_with_index,
                                       flatten,
                                       stats_cumulative_sliding_train,
                                       compute_precision_recall_specificity,
                                       split_dataset_with_date)
from model.dataset import NNStocksDataset
from src.neural_network.model import NeuralNetworkPricePositiveNegative
# from matplotlib import pyplot as plt
# from scipy.stats import norm
# import numpy as np


def train_neural_network(train_dataloader, parameters_model: dict, size_input: int) -> NeuralNetworkPricePositiveNegative:
    """
    Standard training of the neural network.
    :param train_dataloader: pytorch dataloader (wrapper to the training dataset)
    :param parameters_model: dictionary of the parameters of the model
    :param size_input: number of input features
    :return: trained model
    """

    # Instantiate the model
    torch.manual_seed(41)
    model = NeuralNetworkPricePositiveNegative(size_input, 100, 50)

    # Choose loss function
    # requires y.unsqueeze(1) because size must be [size_batch, 1]
    loss_function = torch.nn.BCELoss()
    # loss_func = torch.nn.CrossEntropyLoss()  # prediction: requires vector of probability for each class (FloatTensor); GT: class (LongTensor)
    # requires loss = loss_func(y_predicted, y)
    # Huber Loss for regression
    # loss_func = partial(torch.nn.functional.huber_loss, delta=0.01)

    # Set optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=parameters_model["learning_rate"],
        weight_decay=parameters_model["weight_decay"]
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=parameters_model["learning_rate"])

    # if torch.backends.mps.is_available():  # Train on GPU if possible
    #   mps_device = torch.device("mps")
    #   model.to(mps_device)

    # Conduct training which consists of homing the model in on the best parameters that minimize the loss
    # Epoch: one run through all the training data
    losses = []
    # print(len(train_dataloader.dataset))
    for i in range(parameters_model["epochs"]):
        # Step 1: go forward through the network and get a prediction
        total_loss = 0.
        for x, y in train_dataloader:
            optimizer.zero_grad()
            y_predicted = model(x)
            y = y.unsqueeze(1)
            loss = loss_function(y_predicted, y.to(torch.float32))  # y.type(torch.LongTensor))
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().numpy()
        losses.append(total_loss)
        #print(total_loss)
        # NOTE: accumulated loss increases in case of accumulating training window because there are more and more data for training
        #if i % 100 == 0:
            # print(f"Epoch {i}: loss {total_loss}")
        #    print(total_loss)

    # return trained model (Python's return is always a reference)
    return copy.deepcopy(model)


def train_and_deploy_neural_network(data: pd.DataFrame, predictors: list,
                                    parameters_model: dict,
                                    start_date_training: datetime, end_date_training: datetime,
                                    filename: str,
                                    threshold_probability_positive: float = .6):
    """
    Trains the CNNStocksModule model
    :param data: full dataset
    :param predictors: list of predictors (column names)
    :param parameters_model: dictionary of parameters of the model
    :param start_date_training: initial date for training
    :param end_date_training: final date for training
    :param filename: filename
    :param threshold_probability_positive: probability to accept a positive class as positive
    :return: None
    """
    print("ATTENTION: as of February 2024, training on mps M1 does not converge. CPU must be used!")

    # Put data into GPU if possible (Mac M1: mps)
    torch.manual_seed(41)
    dataloader_kwargs = {}
    #if torch.backends.mps.is_available():
    #   torch.set_default_device("mps")  # Store all training data in the GPU
    #   dataloader_kwargs['generator'] = torch.Generator(device='mps')

    # define train and test dataset
    number_predictors = len(predictors)
    train_dataset_df, test_dataset_df = split_dataset_with_date(data, start_date_training, end_date_training)

    # Turn pandas objects into Pytorch tensor objects and create dataloader
    x_train_tensor, y_train_tensor = (torch.tensor(train_dataset_df[predictors].values),
                                      torch.tensor(train_dataset_df["Target"].values))
    train_dataset = NNStocksDataset(x_train_tensor, y_train_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=parameters_model["batch_size"], shuffle=True, **dataloader_kwargs)

    x_test_tensor, y_test_tensor = (torch.tensor(test_dataset_df[predictors].values),
                                    torch.tensor(test_dataset_df["Target"].values))

    # Train the neural network
    trained_model = train_neural_network(train_dataloader, parameters_model, number_predictors)

    # save model
    pickle.dump(trained_model, open(filename, "wb"))

    # load model
    model_loaded = pickle.load(open(filename, "rb"))

    # Test the trained model
    predictions = model_loaded(x_test_tensor)
    predicted_classes = check_prediction_probability_binary(predictions.squeeze(1).tolist(), threshold_probability_positive)
    # calculate scores
    # PRECISION: ability of the classifier not to label as positive a sample that is negative.
    # When the model predict a positive day, it was right precision% of times
    # SPECIFICITY: ability to predict a negative class correctly.
    # When the model predict a negative day, it was right specificity% of times
    if len(y_test_tensor.tolist()) != len(predicted_classes):
        print("ERROR: size error")
        exit()
    precision, recall, specificity = compute_precision_recall_specificity(y_test_tensor.tolist(), predicted_classes)
    print(f"Precision {precision:.2f}, recall {recall:.2f} specificity {specificity:.2f}")
    print(" ")

    return 0


def train_and_backtest(data: pd.DataFrame, predictors: list, parameters_model: dict,
                       days_initial_train: int = 2500, days_test: int = 250,
                       threshold_probability_positive: float = .6) -> tuple:
    """
    Function to create, train and backtest the model.
    We train for the first days_initial_train days, and we test the following days_test days.
    Then we train for the (days_initial_train + k*days_test) days, and we test the following days_test days.
    NOTE: this is just a backtest to test all the possible successive trainings.
    """

    # Put data into GPU if possible (Mac M1: mps)
    dataloader_kwargs = {}
    # if torch.backends.mps.is_available():
    #   print("MPS available")
    #   torch.set_default_device("mps")  # Store all training data in the GPU
    #   dataloader_kwargs['generator'] = torch.Generator(device='mps')

    score_cumulative_train = {"Target": [], "Prediction": []}
    number_trading_days = data.shape[0]
    number_predictors = len(predictors)

    # Expand the training dataset
    for i in range(days_initial_train, number_trading_days, days_test):
        train_dataset_df, test_dataset_df = split_dataset_with_index(data, 0, i, days_test)

        # Turn pandas objects into Pytorch tensor objects and create dataloader
        x_train_tensor, y_train_tensor = (torch.tensor(train_dataset_df[predictors].values),
                                          torch.tensor(train_dataset_df["Target"].values))
        train_dataset = NNStocksDataset(x_train_tensor, y_train_tensor)
        train_dataloader = DataLoader(train_dataset, batch_size=parameters_model["batch_size"], shuffle=True, **dataloader_kwargs)

        x_test_tensor, y_test_tensor = (torch.tensor(test_dataset_df[predictors].values),
                                        torch.tensor(test_dataset_df["Target"].values))

        # Train the neural network
        trained_model = train_neural_network(train_dataloader, parameters_model, number_predictors)

        # Test the trained model
        with torch.no_grad():
            predictions = trained_model(x_test_tensor)
        predicted_classes = check_prediction_probability_binary(predictions.squeeze(1).tolist(), threshold_probability_positive)
        score_cumulative_train["Target"].append(y_test_tensor.tolist())
        score_cumulative_train["Prediction"].append(predicted_classes)

    # Slide the training windows with a fixed size
    score_sliding_train = {"Target": [], "Prediction": []}
    k = 0
    for i in range(days_initial_train, number_trading_days, days_test):
        start_train_idx = k * days_test
        train_dataset_df, test_dataset_df = split_dataset_with_index(data, start_train_idx, i, days_test)

        # Turn pandas objects into Pytorch tensor objects and create dataloader
        x_train_tensor, y_train_tensor = (torch.tensor(train_dataset_df[predictors].values).float(),
                                          torch.tensor(train_dataset_df["Target"].values).float())
        train_dataset = NNStocksDataset(x_train_tensor, y_train_tensor)
        train_dataloader = DataLoader(train_dataset, batch_size=parameters_model["batch_size"], shuffle=True, **dataloader_kwargs)

        x_test_tensor, y_test_tensor = (torch.tensor(test_dataset_df[predictors].values).float(),
                                        torch.tensor(test_dataset_df["Target"].values).float())

        # Train the neural network
        trained_model = train_neural_network(train_dataloader, parameters_model, number_predictors)

        # Test the trained model
        with torch.no_grad():
            predictions = trained_model(x_test_tensor)
        predicted_classes = check_prediction_probability_binary(predictions.squeeze(1).tolist(), threshold_probability_positive)
        score_sliding_train["Target"].append(y_test_tensor.tolist())
        score_sliding_train["Prediction"].append(predicted_classes)
        k += 1

    return score_cumulative_train, score_sliding_train


def create_and_backtest_neural_network(dataset: pd.DataFrame, predictors_list: list,
                                       parameters_neural_network: dict,
                                       training_days_initial: int = 2500, test_days_step: int = 250,
                                       threshold_probability_positive: float = .6) -> None:
    """
    ADD DESCRIPTION HERE
    :param dataset: full dataset
    :param predictors_list: list of predictors (column names) for the model input
    :param parameters_neural_network: dictionary of parameters of the model
    :param training_days_initial: length of the initial training window
    :param test_days_step: length of the test window
    :param threshold_probability_positive: probability threshold to accept class q as output from the model
    """
    # print("ATTENTION: as of February 2024, training on mps M1 does not converge. CPU must be used!")

    # Backtest the model
    cumulative_training, sliding_training = train_and_backtest(dataset, predictors_list, parameters_neural_network,
                                                               days_initial_train=training_days_initial,
                                                               days_test=test_days_step,
                                                               threshold_probability_positive=threshold_probability_positive)

    # calculate scores
    # PRECISION: ability of the classifier not to label as positive a sample that is negative.
    # When the model predict a positive day, it was right precision% of times
    # SPECIFICITY: ability to predict a negative class correctly.
    # When the model predict a negative day, it was right specificity% of times

    if len(flatten(cumulative_training["Target"])) != len(flatten(cumulative_training["Prediction"])):
        print("ERROR: size error")
        exit()
    print(" ")
    print(predictors_list)
    stats_cumulative_sliding_train(cumulative_training, sliding_training)


def predict(trained_model, x_df: pd.DataFrame) -> pd.Series:
    """
    Generates predictions using a trained model
    :param trained_model: Trained Pytorch model
    :param x_df: Inputs to generate predictions for
    :return: Series containing predictions, with reference dates as indices
    """
    trained_model.eval()

    x_tensor = torch.tensor(x_df.values).float()
    prediction = trained_model(x_tensor)

    return pd.Series(prediction.detach().numpy(), index=x_df.index)


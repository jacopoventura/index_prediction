from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional


def create_lstm_model_regression(sequence_length: int, number_features: int,
                                 number_layers: int = 4, number_neurons: int = 256,
                                 dropout: float = 0.4, bidirectional: bool = True,
                                 loss: str = "mae", optimizer: str = "adam") -> Sequential:
    """
    Create a LSTM model for regression task.
    Notably, the input of the model is a 3D tensor of shape (N, sequence_length, num_features), where N the batch size.
    If a single batch is provided (for example, a single prediction), then the size is (1, sequence_length, num_features).
    Thus, the first cell of the LSTM has to have input_shape=(sequence_length, num_features). In this case, the batch size is any.
    Source: https://www.kaggle.com/code/shivajbd/input-and-output-shape-in-lstm-keras
    :param sequence_length: length of the time sequence
    :param number_features: number of features for each sequence
    :param number_layers: total number of LSTM layers
    :param number_neurons: number of neurons in a single LSTM layer
    :param dropout: dropout rate for each LSTM layer
    :param bidirectional: flag to use a bidirectional layer or a simple forward layer
    :param loss: loss function
    :param optimizer: optimizer
    :return: compiled sequential model
    """

    # In Keras, there are 3 ways to create a NN model.
    # 1. Sequential model (model = Sequential()): easiest way, only sequences of layers and not parallel layers
    # 2. Functional API: x = layer()(x): most popular, allows parallel layers definition
    # 3. Model subclassing
    # In this function, the sequential model is enough because the model is of a simple sequence of layers
    model = Sequential()

    # RNN Bidirectional: adds a new layer of the same cell that allows to process the input in backward direction
    # Bidirectional(LSTM())

    # Dropout layer: given the input ration r, r neurons are randomly selected and ignored during the training.
    # This helps to increase the importance of each single neuron and reduce overfitting
    # Note: dropout is a layer and is applied to the previous layer
    cell = LSTM
    for i in range(number_layers):
        if i == 0:
            # first layer
            if bidirectional:
                model.add(Bidirectional(cell(number_neurons, return_sequences=True), input_shape=(sequence_length, number_features)))
            else:
                model.add(cell(number_neurons, return_sequences=True, input_shape=(sequence_length, number_features)))

        elif i == number_layers - 1:
            # last layer
            last_lstm_layer = cell(number_neurons, return_sequences=False)
            if bidirectional:
                last_lstm_layer = Bidirectional(last_lstm_layer)
            model.add(last_lstm_layer)

        else:
            # hidden layers
            lstm_layer = cell(number_neurons, return_sequences=True)
            if bidirectional:
                lstm_layer = Bidirectional(lstm_layer)
            model.add(lstm_layer)

        # Dropout layer after each layer
        model.add(Dropout(dropout))

    # Dense layer: fully connected neural layer, where each unit of the input is connected to each neuron of the layer.
    # The Dense layer is used to capture complex, non-linear patterns. Usually used as penultimate layer.
    # This is the most basic layer in NN.
    # Dense(10) has 10 neurons, and outputs a (1,10) array.
    # Linear activation: the output is proportional to the input o = K * x
    model.add(Dense(1, activation="linear", use_bias=True))
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
    return model

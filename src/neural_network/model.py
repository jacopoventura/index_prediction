# Copyright (c) 2024 Jacopo Ventura

from torch import nn


class NeuralNetworkPricePositiveNegative(nn.Module):
    # Input layer (N features of the stock data) ->
    # Hidden layer 1 (num of neurons) ->
    # H2 ->
    # output (positive / negative)
    def __init__(self, num_input_features: int, num_neurons_h1: int, num_neurons_h2: int):
        """
        Initializer of the Neural Network model. The init method creates the functions that define each layer of the network.
        :param num_input_features: number of features of the input data
        :param num_neurons_h1: number of neurons of the first hidden layer
        :param num_neurons_h2: number of neurons of the second hidden layer
        """
        super().__init__()  # instantiate the nn.Module() -> class inheritance

        # Rectified Linear Unit: if input <=0, return 0, else return input (ensures non-linearity of the model)
        self.activation = nn.ReLU()

        # Hidden layers of neurons
        self.layer1 = nn.Linear(num_input_features, num_neurons_h1)
        self.layer2 = nn.Linear(num_neurons_h1, num_neurons_h2)
        self.layer3 = nn.Linear(num_neurons_h2, num_neurons_h2)
        #self.layer4 = nn.Linear(100, 50)
        #self.layer5 = nn.Linear(50, 50)

        # The last neuron layer shall output 1 scalar, because we have a scalar target (class 0 or 1)
        self.output = nn.Linear(num_neurons_h2, 1)

        # However, the output of the neuron layer it's simply a number.
        # This number is essentially meaningless information until transformed into a state that can be used to make a classification prediction
        # We transform this output number into a probability by using the Sigmoid function
        # The Sigmoid function returns a float in the range [0; 1].
        # If the input is large, then sigmoid(x) = 1. If the input is very small (negative), then sigmoid(x) = 0. Sigmoid(0) = 0.5
        self.sigmoid = nn.Sigmoid()

        # If the problem is a multiclass classification, then we use softmax instead of Sigmoid()
        # If the problem is a regression problem, then no sigmoid / softmax is needed

        # According to the chosen output, a suitable loss function shall be chosen

    # OUT_CHANNELS = 30  # Number of CNN channels
    # KERNEL_SIZE = 10  # Size of CNN kernel
    # def __init__(self, window_length: int):
    #    super(CNNStocksModule, self).__init__()
    #    assert window_length >= self.KERNEL_SIZE
    #    self.cnn = nn.Conv1d(
    #        1,  # In channel size
    #        self.OUT_CHANNELS,
    #        self.KERNEL_SIZE
    #    )
    #    num_scores = window_length - self.KERNEL_SIZE + 1
    #    # MaxPool kernel size is set such that we only output one value for each row/channel
    #    self.pool = nn.MaxPool1d(num_scores)
    #    self.linear = nn.Linear(self.OUT_CHANNELS, 1, bias=True)

    def forward(self, x):
        """
        The forward defines the forward propagation of the input, which means how a certain input x is processed through the NN layers.
        This function puts all the functions in init into a neural network model
        :param x: input of the neural network
        :return: output of the neural network
        """
        # x = nn.functional.relu(self.fully_connected_layer_1(x))
        # x = nn.functional.relu(self.fully_connected_layer_2(x))
        # x = self.output_layer(x)
        # x = nn.Sigmoid(x)
        # out = self.cnn(x.unsqueeze(1))
        # out = self.pool(out).squeeze()
        # out = torch.softmax(out, dim=1)
        # out = self.linear(out).squeeze()
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        # x = self.activation(self.layer3(x))
        # x = self.activation(self.layer4(x))
        # x = self.activation(self.layer5(x))
        x = self.output(x)
        x = self.sigmoid(x)

        # THE OUTPUT DEPENDS ON THE LOSS FUNCTION (OR VICE VERSA)
        return x

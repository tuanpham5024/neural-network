from abc import ABC

from .Layer import Layer
import numpy as np


class FCLayer(Layer):
    def __init__(self, input_shape, output_shape):
        """
        :param input_shape: (m, n) , example: (1, 3)
        :param output_shape: (m, n), , example: (1, 4)
        """
        # super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.weights = np.random.rand(self.input_shape[1], self.output_shape[1]) - 0.5
        self.bias = np.random.rand(1, output_shape[1]) - 0.5

    def forward_propagation(self, input):
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_propagation(self, ouput_error, learning_rate):
        input_error = np.dot(ouput_error, self.weights.T)
        weights_error = np.dot(self.input.T, ouput_error)

        # update weights with gradient descent
        self.weights -= learning_rate * weights_error
        # update bias with gradient descent
        self.bias -= learning_rate * ouput_error
        return input_error

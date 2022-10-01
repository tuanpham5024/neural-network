from network.Network import Network
from layers.FCLayer import FCLayer
from layers.ActivationLayer import ActivationLayer
import numpy as np


def relu(z):
    """
    :param z: numpy array
    :return: if a <= 0 then 0 else a
    """
    return np.maximum(0, z)


def relu_prime(z):
    """
    :param z: numpy array
    :return: if a <= 0 then 0 else 1
    """
    return np.where(z > 0, 1, 0)


def loss(y_true, y_pred):
    return 0.5 * np.power(y_pred - y_true, 2)


def loss_prime(y_true, y_pred):
    return y_pred - y_true


x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]])

net = Network()
net.add(FCLayer((1, 2), (1, 2)))
net.add(ActivationLayer((1, 2), (1, 2), relu, relu_prime))
net.add(FCLayer((1, 2), (1, 1)))
net.add(ActivationLayer((1, 1), (1, 1), relu, relu_prime))
net.setup_loss(loss, loss_prime)
net.fit(x_train, y_train, 10000, 0.1)

out = net.predict([[0, 1]])
print(out)

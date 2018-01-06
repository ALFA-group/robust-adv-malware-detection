# coding=utf-8
"""
Python module for softmax binary classifier neural network
"""

import torch.nn as nn


def init_weights(net):
    """
    initialize the weights of a network
    :param net:
    :return:
    """

    # init parameters
    def init_module(m):
        if type(m) == nn.Linear:
            nn.init.xavier_normal(m.weight.data)
            nn.init.xavier_uniform(m.bias.data)

    net.apply(init_module)

    return net


def build_ff_classifier(input_size, hidden_1_size, hidden_2_size, hidden_3_size, num_labels=2):
    """
    Constructs a neural net binary classifer
    :param input_size:
    :param hidden_1_size:
    :param hidden_2_size:
    :param hidden_3_size:
    :param num_labels:
    :return:
    """
    net = nn.Sequential(
        nn.Linear(input_size, hidden_1_size),
        nn.ReLU(),
        nn.Linear(hidden_1_size, hidden_2_size),
        nn.ReLU(),
        nn.Linear(hidden_2_size, hidden_3_size),
        nn.ReLU(),
        nn.Linear(hidden_3_size, num_labels),
        nn.LogSoftmax(dim=1))

    return net

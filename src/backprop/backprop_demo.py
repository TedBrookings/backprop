#!/usr/bin/env python

import numpy
from numpy.random import RandomState
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split
from typing import Union

from backprop.network import Network

_demo_problem_num_train_samples: int = 1000
_demo_problem_num_test_samples: int = 100
_demo_num_uninformative_columns: int = 0
_random_state = 0


def demo_backprop(
        num_train_samples: int = _demo_problem_num_train_samples,
        num_test_samples: int = _demo_problem_num_test_samples,
        num_uninformative_columns: int = _demo_num_uninformative_columns,
        random_state: Union[int, None, RandomState] =_random_state
):
    random_state = random_state if isinstance(random_state, RandomState) \
        else RandomState(random_state)
    # make training and test data sets for demo
    inputs_train, inputs_test, outputs_train, outputs_test = make_test_problem(
        n_train_samples=num_train_samples, n_test_samples=num_test_samples,
        n_uninformative=num_uninformative_columns, random_state=random_state
    )
    # build network
    num_inputs = inputs_train.shape[1]
    num_outputs = outputs_train.shape[1]
    num_hidden = 2 * num_inputs * num_outputs
    # make a network with a single hidden layer with num_hidden nodes
    network = Network(num_inputs, num_hidden, num_outputs,
                      random_state=random_state)
    # to make two hidden layers, could do:
    # network = Network(num_inputs, num_hidden, num_hidden, num_outputs)
    # train network on training set
    network.train_online(inputs=inputs_train, correct_outputs=outputs_train)
    # predict results on test set
    predict_test = network.predict(inputs_test)
    # calculate error
    err = ((predict_test - outputs_test)**2).sum(axis=1).mean(axis=0)
    print('Cross-validated error: %.3g' % err)


def make_test_problem(
        n_train_samples: int = _demo_problem_num_train_samples,
        n_test_samples: int = _demo_problem_num_test_samples,
        n_uninformative: int = 0,
        random_state: Union[int, None, RandomState] = _random_state
) -> (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray):
    n_samples = n_train_samples + n_test_samples
    assert n_uninformative >= 0
    n_features = 5 + n_uninformative

    inputs, outputs = make_friedman1(
        n_samples=n_samples, n_features=n_features, random_state=random_state
    )
    if inputs.ndim == 1:
        inputs = numpy.reshape(inputs, inputs.shape + (1,))
    if outputs.ndim == 1:
        outputs = numpy.reshape(outputs, outputs.shape + (1,))

    inputs_train, inputs_test, outputs_train, outputs_test = train_test_split(
        inputs, outputs,
        train_size=n_train_samples, test_size=n_test_samples,
        random_state=random_state
    )

    return inputs_train, inputs_test, outputs_train, outputs_test


if __name__ == "__main__":
    demo_backprop()

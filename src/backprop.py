#!/usr/bin/env python

import numpy
import scipy
from scipy.special import expit
from sklearn.datasets import make_friedman1
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from typing import Callable, Optional
import abc


_default_epoch_learning_rate: float = 0.1
_tol: float = 1.0e-3
_max_epochs_no_improvement: int = 100
_scale_inputs: bool = True
_permute_order: bool = True
_verbose: int = 1
_demo_problem_num_train_samples: int = 1000
_demo_problem_num_test_samples: int = 100
_random_state = 0
_default_hidden_activation_name: str = 'TanhActivation'
_default_output_activation_name: str = 'LinearActivation'


class ActivationFunc(metaclass=abc.ABCMeta):
    __slots__ = tuple()

    @abc.abstractmethod
    def __call__(self, linear_inputs: numpy.ndarray)\
            -> (numpy.ndarray, numpy.ndarray):
        """Should calculate activation and derivative from linear inputs"""
        return numpy.array([]), numpy.array([])

    @classmethod
    def create(cls, *args, **kwargs):
        return cls(*args, **kwargs)


class TwoCallActivationFunc(ActivationFunc):
    __slots__ = ('activation', 'derivative')

    def __init__(
            self,
            activation: Callable[[numpy.array], numpy.array],
            derivative: Callable[[numpy.array], numpy.array]
    ):
        self.activation = activation
        self.derivative = derivative

    def __call__(
            self,
            linear_activation: numpy.ndarray
    ) -> (numpy.ndarray, numpy.ndarray):
        return self.activation(linear_activation), \
               self.derivative(linear_activation)


class LinearActivation(ActivationFunc):
    __slots__ = tuple()

    def __init(self):
        pass

    def __call__(
            self,
            linear_activation: numpy.ndarray
    ) -> (numpy.ndarray, numpy.ndarray):
        return linear_activation, numpy.ones(linear_activation.shape,
                                             dtype=float)


class LinearThresholdActivation(ActivationFunc):
    __slots__ = tuple()

    def __init__(self):
        super().__init__()

    def __call__(
            self,
            linear_activation: numpy.ndarray
    ) -> (numpy.ndarray, numpy.ndarray):
        activation = numpy.where(linear_activation < 0.0, 0.0, linear_activation)
        derivative = numpy.where(linear_activation < 0.0, 0.0, 1.0)
        return activation, derivative


class TanhActivation(ActivationFunc):
    __slots__ = tuple()

    def __call__(
            self,
            linear_activation: numpy.ndarray
    ) -> (numpy.ndarray, numpy.ndarray):
        activation = numpy.tanh(linear_activation)
        derivative = 1.0 - activation * activation
        return activation, derivative


class LogisticActivation(ActivationFunc):
    __slots__ = tuple()

    def __call__(
            self,
            linear_activation: numpy.ndarray
    ) -> (numpy.ndarray, numpy.ndarray):
        activation = expit(linear_activation)
        derivative = activation * (1.0 - activation)
        return activation, derivative


_default_hidden_activation: ActivationFunc \
    = globals()[_default_hidden_activation_name]
_default_output_activation: ActivationFunc \
    = globals()[_default_output_activation_name]


def demo_backprop():
    # make training and test data sets for demo
    inputs_train, inputs_test, outputs_train, outputs_test = make_test_problem()
    # build network
    num_inputs = inputs_train.shape[1]
    num_outputs = outputs_train.shape[1]
    num_hidden = 5 * num_inputs * num_outputs
    network = Network(num_inputs, num_hidden, num_hidden, num_hidden, num_outputs)
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
        random_state: Optional[int] = _random_state
) -> (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray):
    n_samples = n_train_samples + n_test_samples
    assert n_uninformative >= 0
    n_features = 5 + n_uninformative

    inputs, outputs = make_friedman1(
        n_samples=n_samples, n_features=n_features
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


class Layer:
    __slots__ = ('inputs', 'units_out', 'units_deriv', 'activation_func',
                 'weights')

    def __init__(
            self,
            layer_size: int,
            weights: scipy.ndarray,
            activation_func: ActivationFunc,
            *args, **kwargs
    ):
        self.inputs = numpy.empty((weights.shape[0],), dtype=float)
        self.units_out = numpy.empty((layer_size,), dtype=float)
        self.units_deriv = numpy.empty((layer_size,), dtype=float)
        self.activation_func = activation_func.create(*args, **kwargs)
        self.weights = weights

    def __len__(self) -> int:
        return len(self.units_out)

    def forward_prop(
            self,
            inputs: numpy.ndarray
    ) -> numpy.ndarray:
        self.inputs = inputs
        self.units_out, self.units_deriv = self.activation_func(
            inputs.dot(self.weights)
        )
        return self.units_out

    def backward_prop(
            self,
            errors: numpy.ndarray
    ) -> numpy.ndarray:
        errors *= self.units_deriv
        self.weights -= numpy.outer(self.inputs, errors)
        errors = self.weights.dot(errors)
        return errors


class Network:
    __slots__ = ('layers', 'epoch_learning_rate', '_scaler')

    def __init__(
            self,
            *layer_sizes,
            hidden_activation: ActivationFunc = _default_hidden_activation,
            output_activation: ActivationFunc = _default_output_activation,
            epoch_learning_rate: float = _default_epoch_learning_rate
    ):
        assert len(layer_sizes) >= 2, \
            'Must have at least 2 layers (input and output)'
        def _init_weights(_n_pre, _n_post):
            _s = 2 * scipy.sqrt(6.0 / (_n_pre + _n_post))
            #_w = numpy.random.randn(_n_pre, _n_post) * _s
            _w = numpy.random.uniform(-_s, _s, size=(_n_pre, _n_post))
            return _w
        output_index = len(layer_sizes) - 2
        self.layers = [
            Layer(
                n_post,
                activation_func=hidden_activation if ind < output_index
                    else output_activation,
                weights=_init_weights(n_pre, n_post)
            )
            for ind, (n_pre, n_post) in enumerate(
                zip(layer_sizes[:-1], layer_sizes[1:])
            )
        ]
        self.epoch_learning_rate = epoch_learning_rate
        self._scaler = RobustScaler(with_centering=False, with_scaling=False)

    def forward_prop(
            self,
            inputs: numpy.ndarray
    ) -> numpy.ndarray:
        inputs = inputs.copy()
        for layer in self.layers:
            inputs = layer.forward_prop(inputs)
        return inputs

    def backward_prop(
            self,
            errors: numpy.ndarray,
            learning_rate: float
    ):
        errors = learning_rate * errors.copy()
        for layer in self.layers[::-1]:
            errors = layer.backward_prop(errors)

    def predict(
            self,
            inputs: numpy.ndarray
    ) -> numpy.ndarray:
        inputs = self._scaler.transform(inputs)
        num_rows = inputs.shape[0]
        num_cols = len(self.layers[-1])
        outputs = numpy.empty((num_rows, num_cols), dtype=float)
        for row in range(num_rows):
            outputs[row, :] = self.forward_prop(inputs[row, :])
        return outputs

    def train_online_epoch(
            self,
            inputs: numpy.ndarray,
            correct_outputs: numpy.ndarray,
            permute_order: bool = _permute_order
    ) -> float:
        num_rows = inputs.shape[0]
        learning_rate = self.epoch_learning_rate / num_rows
        if permute_order:
            permutation = numpy.random.permutation(num_rows)
            inputs = inputs.take(permutation, axis=0)
            correct_outputs = correct_outputs.take(permutation, axis=0)
        data_error = 0.0
        for row in range(num_rows):
            outputs = self.forward_prop(inputs[row, :])
            errors = outputs - correct_outputs[row, :]
            data_error += (errors ** 2).sum()
            self.backward_prop(errors, learning_rate=learning_rate)
        return data_error / num_rows

    def train_online(
            self,
            inputs: numpy.ndarray,
            correct_outputs: numpy.ndarray,
            scale_inputs: bool = _scale_inputs,
            permute_order: bool = _permute_order,
            tol: float = _tol,
            max_epochs_no_improvement: int = _max_epochs_no_improvement,
            verbose: int = _verbose,
    ) -> float:
        if scale_inputs:
            self._scaler = RobustScaler().fit(inputs)
            inputs = self._scaler.transform(inputs)
        data_error = self.train_online_epoch(inputs, correct_outputs,
                                             permute_order=permute_order)
        epoch = 1
        if verbose:
            print('Epoch %d error = %.4g' % (epoch, data_error))
        best_error = data_error
        num_epochs_no_improvement = 0
        quit_immediately = tol**2
        while num_epochs_no_improvement < max_epochs_no_improvement \
                and best_error > quit_immediately:
            data_error = self.train_online_epoch(inputs, correct_outputs,
                                                 permute_order=permute_order)
            epoch += 1
            if data_error >= best_error:
                num_epochs_no_improvement += 1
            else:
                if best_error - data_error < tol:
                    num_epochs_no_improvement += 1
                else:
                    num_epochs_no_improvement = 0
                    if verbose:
                        print('Epoch %d error = %.4g' % (epoch, data_error))
                best_error = data_error
        if verbose:
            print('FINAL Epoch %d error = %.4g' % (epoch, data_error))
        return data_error


if __name__ == "__main__":
    demo_backprop()
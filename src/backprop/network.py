import numpy
import scipy
from numpy.random import RandomState
from sklearn.preprocessing import RobustScaler
from typing import Union, Text

from backprop.activation_func import ActivationFunc

_default_epoch_learning_rate: float = 1.0
_tol: float = 1.0e-3
_max_epochs_no_improvement: int = 100
_scale_inputs: bool = True
_permute_order: bool = True
_verbose: int = 1
# set random_state to None to randomize each time, some constant for repeatable
# behavior
_random_state = 0
_hidden_activation: str = 'Tanh'
_output_activation: str = 'Linear'
_weight_initialization: str = 'Glorot'
_valid_weight_initialization = frozenset({'Glorot'})
_use_bias: bool = True


class Layer:
    __slots__ = ('inputs', 'units_out', 'units_deriv', 'activation_func',
                 'weights', 'use_bias', 'prng')

    def __init__(
            self,
            layer_size: int,
            input_layer_size: int,
            activation_func: Union[Text, ActivationFunc],
            weights: Union[Text, scipy.ndarray] = _weight_initialization,
            use_bias: bool = _use_bias,
            random_state: Union[int, None, RandomState] = _random_state,
            *args, **kwargs
    ):
        prng: RandomState = random_state \
            if isinstance(random_state, RandomState) \
            else RandomState(random_state)
        self.use_bias = use_bias
        num_inputs = input_layer_size + 1 if use_bias else input_layer_size
        self.inputs = numpy.ones((num_inputs,), dtype=float)
        self.units_out = numpy.empty((layer_size,), dtype=float)
        self.units_deriv = numpy.empty((layer_size,), dtype=float)
        if isinstance(activation_func, Text):
            self.activation_func = ActivationFunc.factory(
                activation_func, *args, **kwargs
            )
        else:
            self.activation_func = activation_func.create(*args, **kwargs)

        if isinstance(weights, numpy.ndarray):
            assert weights.shape == (num_inputs, layer_size)
            self.weights = weights
        else:
            assert isinstance(weights, Text), \
                'weights must be numpy.ndarray or string denoting a valid'\
                ' initialization strategy'
            assert weights in _valid_weight_initialization, \
                'Unknown weight initialization strategy %s' % weights
            if weights == 'Glorot':
                s = numpy.sqrt(6 / (num_inputs + layer_size))
                self.weights = prng.uniform(
                    -s, s, size=(num_inputs, layer_size)
                )

    def __len__(self) -> int:
        return len(self.units_out)

    def forward_prop(
            self,
            inputs: numpy.ndarray
    ) -> numpy.ndarray:
        self.inputs[:inputs.size] = inputs
        self.units_out, self.units_deriv = self.activation_func(
            self.inputs.dot(self.weights)
        )

        return self.units_out

    def backward_prop(
            self,
            errors: numpy.ndarray
    ) -> numpy.ndarray:
        errors *= self.units_deriv
        self.weights -= numpy.outer(self.inputs, errors)
        if self.use_bias:
            return self.weights[:-1, :].dot(errors)
        else:
            return self.weights.dot(errors)


class Network:
    __slots__ = ('layers', 'scaler', 'prng')

    def __init__(
            self,
            *layer_sizes,
            hidden_activation: Text = _hidden_activation,
            output_activation: Text = _output_activation,
            random_state: Union[int, None, RandomState] = _random_state,
            use_bias: bool = _use_bias
    ):
        self.scaler = RobustScaler(with_centering=False, with_scaling=False)
        self.prng: RandomState = random_state\
            if isinstance(random_state, RandomState)\
            else RandomState(random_state)
        assert len(layer_sizes) >= 2, \
            'Must have at least 2 layers (input and output)'
        output_index = len(layer_sizes) - 2
        self.layers = [
            Layer(
                n_post, n_pre,
                activation_func=hidden_activation if ind < output_index
                    else output_activation,
                random_state=self.prng,
                use_bias=use_bias
            )
            for ind, (n_pre, n_post) in enumerate(
                zip(layer_sizes[:-1], layer_sizes[1:])
            )
        ]

    def predict(
            self,
            inputs: numpy.ndarray,
            scale_inputs: bool = True
    ) -> numpy.ndarray:
        if scale_inputs:
            inputs = self.scaler.transform(inputs)
        num_rows = inputs.shape[0]
        num_cols = len(self.layers[-1])
        outputs = numpy.empty((num_rows, num_cols), dtype=float)
        for row in range(num_rows):
            outputs[row, :] = self._forward_prop(inputs[row, :])
        return outputs

    def train_online(
            self,
            inputs: numpy.ndarray,
            correct_outputs: numpy.ndarray,
            epoch_learning_rate: float = _default_epoch_learning_rate,
            scale_inputs: bool=True,
            permute_order: bool = _permute_order,
            tol: float = _tol,
            max_epochs_no_improvement: int = _max_epochs_no_improvement,
            verbose: int = _verbose,
    ) -> float:
        if scale_inputs:
            self.scaler = RobustScaler().fit(inputs)
            inputs = self.scaler.transform(inputs)
        else:
            self.scaler = RobustScaler(with_centering=False, with_scaling=False)

        num_rows = inputs.shape[0]
        learning_rate = epoch_learning_rate / num_rows
        data_error = self._train_online_epoch(inputs, correct_outputs,
                                              learning_rate=learning_rate,
                                              permute_order=permute_order)
        epoch = 1
        if verbose:
            print('Epoch %d error = %.4g' % (epoch, data_error))
        best_error = data_error
        num_epochs_no_improvement = 0
        quit_immediately = tol**2
        while num_epochs_no_improvement < max_epochs_no_improvement \
                and best_error > quit_immediately:
            data_error = self._train_online_epoch(inputs, correct_outputs,
                                                  learning_rate=learning_rate,
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

    def _forward_prop(
            self,
            inputs: numpy.ndarray
    ) -> numpy.ndarray:
        inputs = inputs.copy()
        for layer in self.layers:
            inputs = layer.forward_prop(inputs)
        return inputs

    def _backward_prop(
            self,
            errors: numpy.ndarray,
            learning_rate: float
    ):
        errors = learning_rate * errors.copy()
        for layer in self.layers[::-1]:
            errors = layer.backward_prop(errors)

    def _train_online_epoch(
            self,
            inputs: numpy.ndarray,
            correct_outputs: numpy.ndarray,
            learning_rate: int,
            permute_order: bool = _permute_order
    ) -> float:
        num_rows = inputs.shape[0]
        if permute_order:
            # scramble the order so each epoch traversal is slightly different
            permutation = self.prng.permutation(num_rows)
            inputs = inputs.take(permutation, axis=0)
            correct_outputs = correct_outputs.take(permutation, axis=0)
        data_error = 0.0
        for row in range(num_rows):
            outputs = self._forward_prop(inputs[row, :])
            errors = outputs - correct_outputs[row, :]
            data_error += (errors ** 2).sum()
            self._backward_prop(errors, learning_rate=learning_rate)
        return data_error / num_rows

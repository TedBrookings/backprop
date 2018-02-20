import numpy
from typing import Callable, Text
import abc
from scipy.special import expit as logistic


class ActivationFunc(metaclass=abc.ABCMeta):
    """
    Abstract base class for activation funcs, with supplied factory. e.g.
        ActivationFunc.factory('Tanh') will return a Tanh ActivationFunc.
    Subclasses just need to implement __call__, which takes linear_inputs
    and returns activation and derivative
    """
    __slots__ = tuple()

    @abc.abstractmethod
    def __call__(self, linear_inputs: numpy.ndarray)\
            -> (numpy.ndarray, numpy.ndarray):
        """Should calculate activation and derivative from linear inputs"""
        return numpy.array([]), numpy.array([])

    @staticmethod
    def factory(
            subclass_name: Text,
            *args,
            **kwargs
    ) -> 'ActivationFunc':
        cls = globals()[subclass_name]
        return cls.create(*args, **kwargs)

    @classmethod
    def create(cls, *args, **kwargs) -> 'ActivationFunc':
        return cls(*args, **kwargs)


class TwoCall(ActivationFunc):
    """
    Simple wrapper class that calls activation and derivative funcs that are user
    supplied
    """
    __slots__ = ('activation', 'derivative')

    def __init__(
            self,
            activation: Callable[[numpy.array], numpy.array],
            derivative: Callable[[numpy.array], numpy.array]
    ):
        super().__init__()
        self.activation = activation
        self.derivative = derivative

    def __call__(
            self,
            linear_activation: numpy.ndarray
    ) -> (numpy.ndarray, numpy.ndarray):
        return self.activation(linear_activation), \
               self.derivative(linear_activation)


class Linear(ActivationFunc):
    """
    Basically a no-op, typically used for output layer on regression problems.
    """
    __slots__ = tuple()

    def __init__(self):
        super().__init__()

    def __call__(
            self,
            linear_activation: numpy.ndarray
    ) -> (numpy.ndarray, numpy.ndarray):
        return linear_activation, numpy.ones(linear_activation.shape,
                                             dtype=float)


class ReLU(ActivationFunc):
    """
    ReLU = Rectified Linear Unit. Rectifies to 0 when input is negative,
    otherwise it's linear.
    """
    __slots__ = tuple()

    def __init__(self):
        super().__init__()

    def __call__(
            self,
            linear_activation: numpy.ndarray
    ) -> (numpy.ndarray, numpy.ndarray):
        derivative = (linear_activation > 0.0).astype('float')
        activation = derivative * linear_activation
        return activation, derivative


class LeakyReLU(ActivationFunc):
    """
    Leaky version of ReLU. Instead of rectifying to 0 for negative input, the
    derivative drops to 0.1. This allows some gradient to exist while still
    maintaining a nonlinearity.
    """
    __slots__ = tuple()

    def __init__(self):
        super().__init__()

    def __call__(
            self,
            linear_activation: numpy.ndarray
    ) -> (numpy.ndarray, numpy.ndarray):
        derivative = numpy.where(linear_activation > 0.0, 1.0, 0.1)
        activation = derivative * linear_activation
        return activation, derivative


class Tanh(ActivationFunc):
    """
    Tanh output unit.
    """
    __slots__ = tuple()

    def __call__(
            self,
            linear_activation: numpy.ndarray
    ) -> (numpy.ndarray, numpy.ndarray):
        super().__init__()
        activation = numpy.tanh(linear_activation)
        derivative = 1.0 - activation * activation
        return activation, derivative


class Logistic(ActivationFunc):
    """
    Logistic output unit.
    """
    __slots__ = tuple()

    def __call__(
            self,
            linear_activation: numpy.ndarray
    ) -> (numpy.ndarray, numpy.ndarray):
        super().__init__()
        activation = logistic(linear_activation)
        derivative = activation * (1.0 - activation)
        return activation, derivative

This code represents an attempt to understand the backpropagation algorithm for feedforward neural networks by writing one. The project goal is to encounter the major problems that feedforward networks encounter, and implement the solutions to those problems. To do so, it is challenged by synthetic data from the sklearn module. This code is **NOT** intended to be competitive in performance with existing neural network libraries, nor to implement a common API.

To try out functionality, run demo.py

This project is licensed under the terms of Apache license 2.0

Problems:
- ReLU activations tend to produce infs and NaNs if the learning rate is too large.

To-Do
- Add docstrings.
- Add inputparser to allow running demo with different options.
- Add support for early stopping
- Add "momentum" term to learning rate.
- Add options for different weight initialization strategies.
- Add additional test problems.
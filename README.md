This code represents an attempt to understand the backpropagation algorithm for feedforward neural networks by writing one. The project goal is to encounter the major problems that feedforward networks encounter, and implement the solutions to those problems. To do so, it is challenged by synthetic data from the sklearn module. This code is **NOT** intended to be competitive in performance with existing neural network libraries, nor to implement a common API.

This project is licensed under the terms of Apache license 2.0

Current Problems
- Networks with tanh and ReLU activation functions don't converge to solutions.
- Convergence is slower than expected.

To-Do
- Add bias and bias weights to layers.
- Add "momentum" term to learning rate.
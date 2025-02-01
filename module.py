import numpy as np
from abc import ABC, abstractmethod

class Module(ABC):
    """Abstract base class for neural network modules c.f. PyTorch nn.Module."""
    def __init__(self):
        self.training = True

    @abstractmethod
    def forward(self, *input):
        pass

    @abstractmethod
    def backward(self, *gradwrtoutput):
        pass

    def __call__(self, *input):
        return self.forward(*input)
    
    
class Parameter:
    """Parameter class for neural network modules."""
    def __init__(self, data: np.ndarray):
        self.data = data
        self.grad = np.zeros_like(data)

    def __iadd__(self, other):
        self.data += other
        return self

    def __isub__(self, other):
        self.data -= other
        return self

    def __imul__(self, other):
        self.data *= other
        return self

    def __itruediv__(self, other):
        self.data /= other
        return self

    def __add__(self, other):
        return self.data + other

    def __sub__(self, other):
        return self.data - other

    def __mul__(self, other):
        return self.data * other

    def __truediv__(self, other):
        return self.data / other

    def __repr__(self):
        return repr(self.data)

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)


class Optimizer(ABC):
    """Abstract base class for optimizers."""
    @abstractmethod
    def step(self):
        pass
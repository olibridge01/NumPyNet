import numpy as np
from abc import ABC, abstractmethod

from module import Module, Parameter, Optimizer

class Linear(Module):
    """Linear layer."""
    def __init__(self, in_dim: int, out_dim: int):
        super(Linear, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.W = Parameter(np.random.randn(in_dim, out_dim))
        self.b = Parameter(np.random.randn(out_dim))
        self.input = None

    def forward(self, x: np.ndarray):
        self.input = x
        return np.dot(x, self.W.data) + self.b.data
    
    def backward(self, gradwrtoutput: np.ndarray):
        self.W.grad += np.dot(self.input.T, gradwrtoutput)
        self.b.grad += np.sum(gradwrtoutput, axis=0)
        return np.dot(gradwrtoutput, self.W.data.T)
    

class ReLU(Module):
    """Rectified Linear Unit (ReLU) activation function."""
    def __init__(self):
        super(ReLU, self).__init__()
        self.input = None

    def forward(self, x: np.ndarray):
        self.input = x
        return np.maximum(x, 0)
    
    def backward(self, gradwrtoutput: np.ndarray):
        return gradwrtoutput * (self.input > 0)
    

class Sequential(Module):
    """Sequential container for modules."""
    def __init__(self, *args: Module):
        super(Sequential, self).__init__()
        self.modules = list(args)

    def forward(self, x: np.ndarray):
        for module in self.modules:
            x = module(x)
        return x
    
    def backward(self, gradwrtoutput: np.ndarray):
        for module in reversed(self.modules):
            gradwrtoutput = module.backward(gradwrtoutput)
        return gradwrtoutput
    

class CrossEntropyLoss(Module):
    """Cross-entropy loss function."""
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.input = None
        self.target = None

    def forward(self, x: np.ndarray, target: np.ndarray):
        # Exp-normalise input
        x = np.exp(x) / np.sum(np.exp(x), axis=1)[:, None]
        self.input = x

        
        # Convert target to one-hot encoding if necessary
        if target.ndim == 1:
            target = np.eye(x.shape[1])[target]
        self.target = target

        return -np.sum(target * np.log(x))
    
    def backward(self, gradwrtoutput: np.ndarray):
        return gradwrtoutput * (self.input - self.target)  


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer."""
    def __init__(self, params: list, lr: float):
        self.params = params
        self.lr = lr

    def step(self):
        for param in self.params:
            param -= self.lr * param.grad
            param.grad = np.zeros_like(param.grad)


class RMSProp(Optimizer):
    """Root Mean Square Propagation optimizer."""
    def __init__(self, params: list, lr: float, beta: float = 0.9):
        self.params = params
        self.lr = lr
        self.beta = beta
        self.eps = 1e-8
        self.s = [np.zeros_like(param.data) for param in self.params]

    def step(self):
        for i, param in enumerate(self.params):
            self.s[i] = self.beta * self.s[i] + (1 - self.beta) * param.grad**2
            param.data -= self.lr * param.grad / (np.sqrt(self.s[i]) + self.eps)
            param.grad = np.zeros_like(param.grad)


class Net(Module):
    """Simple feedforward neural network for MNIST handwritten digit classification."""
    def __init__(self, in_dim: int = 784, out_dim: int = 10, hidden_dims: list = [128]):
        super(Net, self).__init__()
        
        # Define layers
        self.layers = Sequential(Linear(in_dim, hidden_dims[0]), ReLU())
        for i in range(len(hidden_dims) - 1):
            self.layers.modules.append(Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.layers.modules.append(ReLU())
        self.layers.modules.append(Linear(hidden_dims[-1], out_dim))

        self.loss = CrossEntropyLoss()
        # Get self.params, excluding those of ReLU
        self.params = [param for module in self.layers.modules for param in module.__dict__.values() if isinstance(param, Parameter)]
        
    def forward(self, x: np.ndarray):
        return self.layers(x)

    def backward(self, gradwrtoutput: np.ndarray):
        return self.layers.backward(gradwrtoutput)
    
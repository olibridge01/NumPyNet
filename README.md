[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)](#)
[![NumPy](https://img.shields.io/badge/NumPy-4DABCF?logo=numpy&logoColor=fff)](#)

# Neural Networks with NumPy
This repository contains a NumPy implementation of a feedforward neural network for training an MNIST handwritten digit classifier. 

## Directory Structure
```
NumPyNet/
├── data/
│   └── (save directory for MNIST)
├── dataset.py (dataset and dataloader classes)
├── model.py (neural network layers)
├── module.py (module/optimizer base classes)
└── train.ipynb (training script notebook)
```

## Model
The model consists of a simple three-layer network with a ReLU activation function. The network is trained using the cross-entropy loss function and the RMSProp optimizer. Below is a diagram showing the model architecture and how the hidden layer values are computed.

<p align="center">
  <img src="https://github.com/user-attachments/assets/798ee0bf-d49c-4ece-bfd8-b209e3afff4a" width="30%"/>
  <img src="https://github.com/user-attachments/assets/2e51df8a-f655-4db0-aad4-603f7c7306fc" width="64%"/>
</p>

We start with a flattened image vector $\mathbf{x} \in \mathbb{R}^{d}$, which is passed through the first layer (with a ReLU activation) to produce a hidden layer $\mathbf{z}^{(1)} \in \mathbb{R}^{h}$. The hidden layer is then passed through the second layer to produce a logit vector $\mathbf{z}^{(2)} \in \mathbb{R}^{K}$, which corresponds to a value for each of the $K$ classes. The logit vector is subsequently passed through a softmax function to produce a probability distribution over the classes. Assuming now that we have a batch of images, $\mathbf{X} \in \mathbb{R}^{N \times d}$, the forward pass can be written as:

$$
\begin{align*}
\mathbf{Z}^{(1)} &= \sigma(\mathbf{X} \mathbf{W}^{(0)} + \mathbf{b}^{(0)}) \in \mathbb{R}^{N \times h} \\
\\
\mathbf{Z}^{(2)} &= \mathbf{Z}^{(1)} \mathbf{W}^{(1)} + \mathbf{b}^{(1)} \in \mathbb{R}^{N \times K} \\
\\
\mathbf{P} &= \text{softmax}(\mathbf{Z}^{(2)}) \in \mathbb{R}^{N \times K} \\
\\
\mathcal{L} &= -\frac{1}{N} \sum_{n=1}^{N} \sum_{k=1}^{K} Y_{nk} \log(P_{nk})
\end{align*}
$$

where $\sigma(\cdot) = \text{ReLU}(\cdot)$ is the activation function, and $\{\mathbf{W}^{(0)}, \mathbf{W}^{(1)}, \mathbf{b}^{(0)}, \mathbf{b}^{(1)}\}$ are the model parameters. The loss function $\mathcal{L}$ is the cross-entropy loss, and $Y_{nk}$ is the one-hot encoded label for the $n$-th sample in the batch. For the backward pass, we compute

$$
\begin{align*}
\frac{\partial \mathcal{L}}{\partial \mathbf{Z}^{(2)}} &= \mathbf{P} - \mathbf{Y} \in \mathbb{R}^{N \times K} \\
\\
\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(1)}} &= \mathbf{Z}^{(1)T} \left( \frac{\partial \mathcal{L}}{\partial \mathbf{Z}^{(2)}} \right) \in \mathbb{R}^{h \times K} \\
\\
\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(1)}} &= \mathbf{1}^{T} \left( \frac{\partial \mathcal{L}}{\partial \mathbf{Z}^{(2)}} \right) \in \mathbb{R}^{K} \\
\\
\frac{\partial \mathcal{L}}{\partial \mathbf{Z}^{(1)}} &= \left( \frac{\partial \mathcal{L}}{\partial \mathbf{Z}^{(2)}} \right) \mathbf{W}^{(1)T} \in \mathbb{R}^{N \times h} \\
\\
\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(0)}} &= \mathbf{X}^{T} \left( \frac{\partial \mathcal{L}}{\partial \mathbf{Z}^{(1)}} \right) \in \mathbb{R}^{d \times h} \\
\\
\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(0)}} &= \mathbf{1}^{T} \left( \frac{\partial \mathcal{L}}{\partial \mathbf{Z}^{(1)}} \right) \in \mathbb{R}^{h}
\end{align*}
$$

where $\mathbf{1} \in \mathbb{R}^{N}$ is a vector of ones. These gradients are propagated backwards (hence backward pass) via the chain rule to compute the gradients with respect to the model parameters. The gradients are then used to update the model parameters using the RMSProp optimizer.





## Model Performance
Below are training loss and accuracy plots, along with a sample of predictions (including logit histograms) on the MNIST test set. Code for plotting both figures is located in the notebook `train.ipynb`.
<!-- <p align="center">
  <img src="https://github.com/user-attachments/assets/1fd51329-884b-443e-a4c5-67a4d3bf3f7d" width="65%"/>
</p> -->

![mnist_loss_acc](https://github.com/user-attachments/assets/1fd51329-884b-443e-a4c5-67a4d3bf3f7d)
![mnist_predictions](https://github.com/user-attachments/assets/91333c90-b6c5-41fe-85c6-c36318330d5e)
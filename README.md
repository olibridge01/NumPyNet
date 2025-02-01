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
The model consists of a simple three-layer network with a ReLU activation function. The network is trained using the cross-entropy loss function and the RMSProp optimizer.

## Model Performance
Below are training loss and accuracy plots, along with a sample of predictions (including logit histograms) on the MNIST test set. Code for plotting both figures is located in the notebook `train.ipynb`.
<!-- <p align="center">
  <img src="https://github.com/user-attachments/assets/1fd51329-884b-443e-a4c5-67a4d3bf3f7d" width="65%"/>
</p> -->

![mnist_loss_acc](https://github.com/user-attachments/assets/1fd51329-884b-443e-a4c5-67a4d3bf3f7d)
![mnist_predictions](https://github.com/user-attachments/assets/91333c90-b6c5-41fe-85c6-c36318330d5e)
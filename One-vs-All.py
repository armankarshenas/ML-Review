from Data_cleaning import feature_matrix_with_normalization,observation_labels
import torch
import random
from tqdm import tqdm
import numpy as np
from torchvision import datasets

data_train = datasets.MNIST(root=".",download=True, train=True)
data_test = datasets.MNIST(root=".",download=True,train=False)

X_train = feature_matrix_with_normalization(data_train)
X_test = feature_matrix_with_normalization(data_test)

y_train = observation_labels(data_train)
y_test = observation_labels(data_test)
y_train = np.reshape(y_train,[len(y_train),1])
y_test = np.reshape(y_test,[len(y_test),1])


n_models = len(np.unique(y_train))
models = []

def Sigmoid(x):
    sig = np.exp(-x)
    sig = 1/(1+sig)
    return sig

def BinaryLoss(y,y_hat):
    loss = -np.multiply(y,np.log(y_hat+1e-4)) - np.multiply((1-y),np.log(1-y_hat+1e-4))
    loss = np.sum(loss)
    loss = loss/len(y)
    return loss

def BinaryLossDerivative(y,y_hat,X):
    der = np.multiply(y_hat,(1-y_hat))
    der = np.multiply((np.divide(y,y_hat+1e-4) - np.divide((1-y),(1-y_hat+1e-4))),der)
    der = np.matmul(np.transpose(X),der)
    der = der/len(y)
    return der

def train_one_vs_all(label,X):
    m = len(X[0][:])
    w = np.random.normal(0,1,[m,1])
    cost = []
    max_iteration = 1e3
    alpha = 0.01
    z = np.matmul(X, w)
    y_pred = Sigmoid(z)
    cost.append(BinaryLoss(label,y_pred))
    for i in tqdm(range(int(max_iteration))):
        derivative = BinaryLossDerivative(label,y_pred,X)
        w = w - alpha*derivative
        y_pred = Sigmoid(np.matmul(X,w))
        loss = BinaryLoss(label,y_pred)
        cost.append(loss)
        if abs(cost[-1] - cost[-1]) <= 0.001:
            return w
    return w


for digit in (np.sort(np.unique(y_train))):
    cond_idx = y_train == digit
    y_local = np.array(cond_idx).astype(int)
    model_local = train_one_vs_all(y_local,X_train)
    models.append(model_local)

print(np.shape(models))

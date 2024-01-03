from Data_cleaning import feature_matrix_with_normalization, observation_labels
from torchvision import datasets
from tqdm import tqdm
import numpy as np
import torch
from torch import nn

data_train = datasets.MNIST(".",train=True,download=True)
data_test = datasets.MNIST(".",train=False,download=True)

print("The datasets shapes are: ",np.shape(data_train),np.shape(data_test))
X_train = feature_matrix_with_normalization(data_train)
X_test  = feature_matrix_with_normalization(data_test)
X_train = torch.tensor(X_train).type(torch.float)
X_test = torch.tensor(X_test).type(torch.float)
print("The shape of the processed feature matrix is: ",np.shape(X_train),np.shape(X_test))
label_train = observation_labels(data_train)
label_test = observation_labels(data_test)
label_train = torch.tensor(label_train).type(torch.float)
label_test = torch.tensor(label_test).type(torch.float)
print("The shape of the processed label vector is: ",np.shape(label_test),np.shape(label_train),label_test.type())


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(785,1024),
            nn.ReLU(),
            nn.Linear(1024,256),
            nn.ReLU(),
            nn.Linear(256,10))
    def forward(self,x):
        return self.layers(x)

network = MLP()
network.to(device)
print(network)

# Write our training loop
loss_training = []
acc_training = []
verbose = 200
max_iteration = int(1e4)
lr = 1e-4
optimizer = torch.optim.SGD(network.parameters(), lr=lr)
loss_fn = torch.nn.CrossEntropyLoss()

for i in tqdm(range(max_iteration)):
    network.train()
    X = X_train.to(device)
    Y = label_train.to(device)
    optimizer.zero_grad()
    pred = network(X)
    loss = loss_fn(pred, Y)
    loss.backward()
    optimizer.step()
    loss_training.append(loss.item())
    network.eval()
    num_correct = (pred.argmax(1) == Y).type(torch.float).sum().item()
    acc_training.append(num_correct / len(Y))
    if i%verbose == 0:
        print("Iteration {i}, loss = {loss} and accuracy = {acc}",i=i,loss=loss_training[-1],acc = acc_training[-1])




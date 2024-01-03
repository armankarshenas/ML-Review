from Data_cleaning import feature_matrix_with_normalization, observation_labels,observation_labels_binary
from torchvision import datasets
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
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
print("The shape of the processed label vector is: ",np.shape(label_train),np.shape(label_test))

data_train = TensorDataset(X_train,label_train)
data_test = TensorDataset(X_test,label_test)

print("Training data is {train_data} and testing data is {testing_data}".format(train_data = data_train,testing_data=data_test))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
training_loader = DataLoader(data_train,batch_size=32,shuffle=True)
testing_loader = DataLoader(data_test,batch_size=len(data_test))

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
epochs = int(10)
lr = 1e-4
optimizer = torch.optim.SGD(network.parameters(), lr=lr)
loss_fn = torch.nn.CrossEntropyLoss()
for epoch in tqdm(range(epochs)):
    for batch, (x,y) in (enumerate(training_loader)):
        network.train()
        X = x.to(device)
        Y = y.to(device)
        optimizer.zero_grad()
        pred = network(X)
        #print(pred.type(torch.long),Y.type(torch.long))
        loss = loss_fn(pred, Y.type(torch.long))
        loss.backward()
        optimizer.step()
        loss_training.append(loss.item())
        network.eval()
        num_correct = (pred.argmax(1) == Y).type(torch.float).sum().item()
        acc_training.append(num_correct / len(Y))
        if batch%verbose == 0:
            print("batch {b}, loss = {loss} and accuracy = {acc}".format(b=batch,loss=loss_training[-1],acc = acc_training[-1]))




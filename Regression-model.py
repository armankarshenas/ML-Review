import numpy as np
from tqdm import tqdm
import pandas as pd
from matplotlib import pyplot as plt
# First let's load the dataset from example 1 using pandas
dt = pd.read_csv("Regression-data/ex1data1.txt",names=["x","y"])
print(dt.head())
x = np.array(dt["x"])
y = np.array(dt["y"])
print("x shape and y shape are:",np.shape(x),np.shape(y))

# Now let's define a model and cost function

def cost(X,w,values):
    mt =np.matmul(X,w) - values
    return np.matmul(np.transpose(mt),mt)

def derivative(X,w,values):
    mt = np.matmul(np.transpose(X),X)
    mt = np.matmul(mt,w) - np.matmul(np.transpose(X),values)
    return 2*mt
def preprocess(X,y,m):
    n = len(X)
    X_processed = np.ones([n,int(m+1)])
    if m==1:
        X_processed[:,1] = ((X - np.mean(X))/np.std(X))
    else:
        for i in range(1,m+1):
            x_local = X[:,i]
            mu = np.mean(x_local)
            std = np.std(x_local)
            X_processed[:,i] = (x_local-mu)/std
    y = np.reshape(y,[len(y),1])
    return X_processed,y
def initialize_parameters(X,m):
    return np.random.normal(0,1,[m+1,1])

def train(X,values, m, max_iteration=int(1e4),threshold = 1e-4,alpha=0.01):
    w = initialize_parameters(X,m)
    cost_arr = []
    cost_arr.append(cost(X,w,values))
    for i in tqdm(range(int(max_iteration))):
        der = derivative(X,w,values)
        w = w - alpha*der
        cost_arr.append(cost(X,w,values))
        if abs(cost_arr[-1]-cost_arr[-2]) < threshold:
            return w,cost_arr
    return w,cost_arr

def alg_solution(X,values):
    mt = np.matmul(np.transpose(X),X)
    mt = np.linalg.inv(mt)
    mt = np.matmul(mt,np.transpose(X))
    mt = np.matmul(mt,values)
    return mt

def visualize(w,w_alg,X,values,m):
    if m>1:
        print("The data cannot be visualized! ")
    if m == 1:
        y_pred = np.matmul(X,w)
        y_pred_alg = np.matmul(X,w_alg)
        x = X[:,1]
        plt.scatter(x,values)
        plt.plot(x,y_pred,'r')
        plt.plot(x,y_pred_alg,'b')
        plt.show()
# Let's try the code for the first model
X,y = preprocess(x,y,1)
print("X shape and y shape are:",np.shape(X),np.shape(y))

w,loss = train(X,y,1,max_iteration=10000,threshold=1e-5)
w_alg = alg_solution(X,y)
str_to_print = "The gradient descent method result is {gd} and the algebraic solution is {alg}".format(gd = w,alg=w_alg)
print(str_to_print)
plt.plot(np.reshape(loss,[len(loss),1]))
plt.xlabel("iteration")
plt.ylabel("loss")
plt.show()
visualize(w,w_alg,X,y,1)






import numpy as np
import torch
from torchvision import datasets
from matplotlib import pyplot as plt
from tqdm import tqdm

def feature_matrix(dataset):
    n = len(dataset)
    pic_template = np.array(dataset[0][0])
    X = np.ones([n,len(pic_template)*len(pic_template[0])+1])
    str_to_print = "observations: {n}, Pic dimensions: {len_1},{len_2}, X dimensions: {X}".format(n = n,len_1 =len(pic_template),len_2 = len(pic_template[0]),X=np.shape(X))
    print(str_to_print)
    for i,ent in enumerate(tqdm(dataset)):
        pic = np.array(ent[0])
        X[i][1:] = np.reshape(pic,[1,len(pic_template)*len(pic_template[0])])
    return X+1

def feature_matrix_with_normalization(dataset):
    n = len(dataset)
    pic_template = np.array(dataset[0][0])
    X = np.ones([n,len(pic_template)*len(pic_template[0])+1])
    str_to_print = "observations: {n}, Pic dimensions: {len_1},{len_2}, X dimensions: {X}".format(n = n,len_1 =len(pic_template),len_2 = len(pic_template[0]),X=np.shape(X))
    print(str_to_print)
    for i,ent in enumerate(tqdm(dataset)):
        pic = np.array(ent[0])
        X[i][1:] = np.reshape(pic,[1,len(pic_template)*len(pic_template[0])])
    X = X + np.random.randint(0,5,[len(X),len(X[0])])
    for i in range(len(X[0])):
        mu = np.mean(X[:,i])
        std = np.std(X[:,i])
        X[:,i] = (X[:,i] - mu)/std
    return X

def observation_labels(dataset):
    label = [ent[1] for ent in dataset]
    label = np.array(label)
    label = np.reshape(label,[len(label),1])
    return label




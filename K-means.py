import numpy as np
import random
from matplotlib import pyplot as plt
from tqdm import tqdm


def initialize_points(n):
    x = np.random.normal(0, 10, [n, 1])
    y = np.random.normal(1, 5, [n, 1])
    #plt.scatter(x,y)
    #plt.show()
    return x,y


def Euclidean_distance(x1,x2,y1,y2):
    return np.sqrt((x1-x2)**2+(y1-y2)**2)


def initialize_centroids(k,x,y):
    len_partition = np.floor(len(x)/k)
    centroids = []
    for i in range(int(k)):
        local_idx = np.random.randint(len(x),size = int(len_partition))
        print(local_idx)
        x_local = x[local_idx]
        y_local = y[local_idx]
        centroids.append([np.mean(x_local),np.mean(y_local)])
    return centroids


def assign_labels(centroids,x,y,k):
    labels = np.zeros_like(x)
    for i in range(len(x)):
        distances = []
        for j in range(int(k)):
            distances.append(Euclidean_distance(x[i],centroids[j][0],y[i],centroids[j][1]))
        labels[i] = np.argmin(distances)
    return labels


def update_centroids(x,y,k,labels):
    new_centroids = []
    for i in range(int(k)):
        cond_idx = labels == i
        x_local = x[cond_idx]
        y_local = y[cond_idx]
        x_new = np.mean(x_local)
        y_new = np.mean(y_local)
        new_centroids.append([x_new,y_new])
    return new_centroids


def centroid_distance(centroid1,centroid2):
    dist = 0
    for i in range(len(centroid1)):
        dist += Euclidean_distance(centroid1[i][0],centroid2[i][0],centroid1[i][1],centroid2[i][1])
    return dist


# Writing the main body of the code here
k = 5
n = 200
max_iteration = int(1e4)
x,y = initialize_points(n)
centroids = initialize_centroids(k,x,y)
for i in tqdm(range(int(max_iteration))):
    labels = assign_labels(centroids,x,y,k)
    new_centroids = update_centroids(x,y,k,labels)
    dst = centroid_distance(new_centroids,centroids)
    if dst > 0.01:
        centroids = new_centroids
    else:
        break
fig, ax = plt.subplots()
for i in range(int(k)):
    idx = labels == i
    x_local = x[idx]
    y_local = y[idx]
    ax.scatter(x_local,y_local,label=i)
ax.legend()
plt.show()

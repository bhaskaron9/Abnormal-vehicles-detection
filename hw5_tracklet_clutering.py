"""
This is a dummy file for HW5 of CSE353 Machine Learning, Fall 2020
You need to provide implementation for this file

By: Minh Hoai Nguyen (minhhoai@cs.stonybrook.edu)
Created: 26-Oct-2020
Last modified: 26-Oct-2020
"""

import random
import numpy as np
import math
from scipy.spatial.distance import cdist


class TrackletClustering(object):
    """
    You need to implement the methods of this class.
    Do not change the signatures of the methods
    """
    db = []
    centroids = []
    def __init__(self, num_cluster):
        self.num_cluster = num_cluster

    def add_tracklet(self, tracklet):
        "Add a new tracklet into the database"
        self.db.append(tracklet)
    def k_means(self, k):
        X = []
        for i in self.db:
            ox1=(i['tracks'][0][1]+i['tracks'][0][3])/2.0
            oy1=(i['tracks'][0][2]+i['tracks'][0][4])/2.0
            ox2=(i['tracks'][-1][1]+i['tracks'][-1][3])/2.0
            oy2=(i['tracks'][-1][2]+i['tracks'][-1][4])/2.0
            nrm = np.linalg.norm(np.array([ox1,oy1])-np.array([ox2,oy2]))
            if nrm == 0:
                X.append([0.0,0.0])
            else:
                direction = [ox1 - ox2 / nrm+1e-10, oy1-oy2 / nrm+1e-10]
                cd = [direction[0] * math.sqrt(2), direction[1] * math.sqrt(2)]
                X.append(cd)
        X = np.array(X)
        ind = np.random.permutation(X.shape[0])
        ind = ind[:k]
        centroids = X[ind]
        c1 = np.zeros(centroids.shape)
        while True:
            c1 = np.zeros(centroids.shape)
            d = cdist( centroids, X, 'euclidean')
            nc = np.argmin(d,axis = 0)
            for i in range(k):
                c1[i] = np.mean(X[np.where(nc==i)],axis = 0)
            if np.array_equal(centroids, c1):
                break;
            centroids = c1
        return centroids

    def build_clustering_model(self):
        "Perform clustering algorithm"
        self.centroids = self.k_means(self.num_cluster)
        return self.centroids

    def get_cluster_id(self, tracklet):
        """
        Assign the cluster ID for a tracklet. This funciton must return a non-negative integer <= num_cluster
        It is possible to return value 0, but it is reserved for special category of abnormal behavior (for Question 2.3)
        """
        i=tracklet
        ox1=(i['tracks'][0][1]+i['tracks'][0][3])/2.0
        oy1=(i['tracks'][0][2]+i['tracks'][0][4])/2.0
        ox2=(i['tracks'][-1][1]+i['tracks'][-1][3])/2.0
        oy2=(i['tracks'][-1][2]+i['tracks'][-1][4])/2.0
        nrm = np.linalg.norm(np.array([ox1,oy1])-np.array([ox2,oy2]))
        if nrm == 0:
                X_test = [0.0,0.0]
        else: 
            direction = [ox1 - ox2 / nrm+1e-10, oy1-oy2 / nrm+1e-10]
            cd = [direction[0] * math.sqrt(2), direction[1] * math.sqrt(2)]
            X_test = cd
        #print("X_test",X_test)
#         euclid_dis=np.array(self.calculateEucleadian([X_test],self.unique_X_centroids,"euclidean"))
        #print(":euclid_dis:",euclid_dis)
#         arg_sort=euclid_dis.argsort()

#         self.test_plot_data.append([X_test[0],X_test[1],arg_sort[0][0]+1])
        #print(":arg_sort:",arg_sort)
        #print(arg_sort[0][0]+1)
#         print("bhas",self.centroids)
        d = cdist( [X_test], self.centroids, 'euclidean')
        ind = np.argmin(d,axis = 1)
        return int(ind[0])+1
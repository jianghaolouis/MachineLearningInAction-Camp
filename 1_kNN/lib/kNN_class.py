#!/usr/bin/env python
# -*-coding: utf-8 -*-

'''
Reforming kNN function, which is inspired by scikit-learn pyhton lib 
@ Author: jianghaolouis

'''

import numpy as np
import operator                          #for sorting
import matplotlib.pyplot as plt
import os

class kNN(object): 

    '''
    K nearest neighbor algorithm and Euclidian distance is chosen as the metric. 

    Parameter:
    ----------
    inX: array.
        sample waiting to be classified.
    dataset: array.
        training set.
    labels: vector
    K: parameter of nb_neighbor

    Results:
    --------
    sortedClassCount[0][0] : one element of array
        the prediction of label
    '''
    
    def __init__(self, neighbor = 3):
        self._k = neighbor
        
    def fit(self, feature, label):
        self._feature = feature
        self._label = label
        return self
    
    
    def predict(self, inX):
    #distance calc
        datasetSize = self._feature.shape[0]        #shape returns the dim of array,  (n , m) first indice 0 is dim n
        rep_inX = np.tile(inX, (datasetSize,1))     #tile is repeating the array inX to (n,m) times, (n,1) means copy into n time in row 
        diff_Mat = (rep_inX - self._feature)**2
        diff_eucl = diff_Mat.sum(axis=1)            #np.sum  axis shows the direction: axis= 0, adding each column, axis=1, adding each row
        dist_eucl =  diff_eucl**0.5

        #voting with lowest k distance
        sortedDistIndice = dist_eucl.argsort()      #return the index of sorted array from least to largest 
        classCount = {}                             #define a dict{}
    
        for i in range(self._k):
            voteLable = self._label[sortedDistIndice[i]]
            classCount[voteLable] = classCount.get(voteLable,0) + 1     #update the classCount .get  search in dict{} , if value not found, return 0 (can be replaced with other value)

        # sort dictionary
        sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
        #take the classCount dict and decompose it into a list of tuples and then sort the tuples by the second item using itemgetter
        # the order is reversed so the result from the largest to least, finally return the label of the item the most frequent
        return sortedClassCount[0][0]               #k = 3, result is [('b', 2), ('a', 1)] take the first label.
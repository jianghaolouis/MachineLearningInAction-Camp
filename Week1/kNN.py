#!/usr/bin/env python
# -*-coding: utf-8 -*-

'''
09/28/2018
My First Python Program
kNN: k nearest neighbor
@author: jianghaolouis
'''

import numpy as np
import operator                          #for sorting
import matplotlib.pyplot as plt

def CreateDataset():
    _group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    _labels = ['a','a','b','b']
    return _group, _labels


def file2mat(fileName, delimeter = '\t') :
    '''
    output feature_array, label_Vector
    '''
    #get number of lines in file
    
    with open(fileName,'r') as fr:
        raw_data = fr.readlines()
        nb_row = len(raw_data)
        nb_col = len(raw_data[0].split(delimeter))
        #initiate Numpy matrix to return
        return_array = np.empty((nb_row,nb_col - 1))
        label_Vector = []

        #parse line to a list
        love_dict = {'largeDoses':3,'smallDoses':2, 'didntLike':1}
        index = 0

        for line in raw_data:
            line = line.strip()                             #strip(), take all the char btw the given char in the str, if '', then take the whole line
            list_By_line = line.split(delimeter)            #element delimited by the tab char
            return_array[index,:] = list_By_line[0:3]
            if(list_By_line[-1].isdigit()):                 #test if the last columne is digital value. create a dict to transfer str to digit
                label_Vector.append(int(list_By_line[-1]))
            else:
                label_Vector.append(love_dict.get(list_By_line[-1]))
            index += 1

    return return_array, np.array(label_Vector)

def load_image(filepath):
    '''
    load data from a image-like file

    Parameters:
    -----------
    fname: str or path-like object( absolute or current working dir)

    Results:
    numpy feature: array.
        the feature vector with the shape (n_features,) 
    '''
    with open(filepath,'r') as file:
        raw_image = [row.strip() for row in file.readlines()]
        raw_image = np.array(list(''.join(raw_image)),dtype = 'i4')     # i4 means int32
    return raw_image

def image_features(folder):
    '''
    get all digit images into a numpy array which serves as the feature array of kNN

    Parameters:
    -----------
    folder path : str or working path dir
        the name of folder where the images are stored.
    Return
    '''
    pass

def img2vec(filename):
    with open(filename,'r') as fr:
        raw_image = fr.readlines()                     # text is a string format
        vec = []
        for raw in raw_image :
            raw = raw.strip()
            vec.append(np.array(list(''.join(raw)),dtype = 'i4'))
    return vec


def kNN_Classify0(inX, dataset, labels, k):
    '''
    K nearest neighbor algorithm

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

    #distance calc
    datasetSize = dataset.shape[0]              #shape returns the dim of array,  (n , m) first indice 0 is dim n
    rep_inX = np.tile(inX, (datasetSize,1))     #tile is repeating the array inX to (n,m) times, (n,1) means copy into n time in row 
    diff_Mat = (rep_inX - dataset)**2
    diff_eucl = diff_Mat.sum(axis=1)            #np.sum  axis shows the direction: axis= 0, adding each column, axis=1, adding each row
    dist_eucl =  diff_eucl**0.5

    #voting with lowest k distance
    sortedDistIndice = dist_eucl.argsort()      #return the index of sorted array from least to largest 
    classCount = {}                             #define a dict{}
   
    for i in range(k):
        voteLable = labels[sortedDistIndice[i]]
        classCount[voteLable] = classCount.get(voteLable,0) + 1     #update the classCount .get  search in dict{} , if value not found, return 0 (can be replaced with other value)

    # sort dictionary
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    #take the classCount dict and decompose it into a list of tuples and then sort the tuples by the second item using itemgetter
    # the order is reversed so the result from the largest to least, finally return the label of the item the most frequent
    return sortedClassCount[0][0]               #k = 3, result is [('b', 2), ('a', 1)] take the first label.

def data_Norm(dataSet):
    '''
    output normalized vector, min_vector and d_interval
    '''
    d_min = dataSet.min(0)                       # 0 means the values in each colume
    d_max = dataSet.max(0)
    d_interval = d_max - d_min
    d_Norm = np.empty(np.shape(dataSet))
    d_Norm = (dataSet - np.tile(d_min,(dataSet.shape[0],1))) / np.tile(d_interval,(dataSet.shape[0],1)) 
    return d_Norm, d_min, d_interval


def Split_train_set(X, y, ratio):
    '''
    creating a training set and test set with given ratio
    randomly choose ratio * len(X) sample as training set. 

    Parameters:
    -----------
    X : array.
        Feature array with the shape(n_sample, n_feature)
    y : array.
        Label array with the shape(n_samples,)
    ratio : float
        The ratio of n_training_set and n_test_set

    Result:
    -------
    x_train : array.
        Training feature array. 
    y_train : array.
        Training label
    x_test : array.
        Testing feature array
    y_test : array.
        Testing label
    '''
    permuted_indices = np.random.permutation(len(X))            #take the indice and get values later
    nb_test= int(ratio * len(X))
    test_indice = permuted_indices[:nb_test]                    #take the first nb_test element, per[nb_test] isn't included 
    train_indice = permuted_indices[nb_test:]
    #take the relavent row
    x_test = X[test_indice,:]
    y_test = y[test_indice]
    x_train = X[train_indice,:]
    y_train = y[test_indice]

    return x_test,y_test,x_train,y_train
    
def ErrorTest(ratio,filename,k, method = 'kNN'):
    '''
    using the given algorithm to test the error rate of prediction

    Parameter:
    ----------
    ratio: float.
        division of testing set and training set
    filename: array.
        the row data
    k : int
        number of nearest neighborhood
    
    Results:
    --------
    errorRate:float.
        the error rate of prediction using given method.
    '''
    _X_raw, _y = file2mat(filename)
    _X, d_min, d_interval = data_Norm(_X_raw)
    x_test,y_test,x_train,y_train = Split_train_set(_X,_y,ratio)
    err_Count = 0.0
    for i in range(len(x_test)):
        Predict_rs = kNN_Classify0(x_test[i], _X, _y, k)
        # print( "the classifer came back with : %s, the real answer is :%s" %(Predict_rs,y_test[i]))
        if (Predict_rs != y_test[i]): err_Count += 1.0
    errorRate = err_Count/float(len(x_test))        
    #print("the total error rate is : %f " % errorRate)
    return errorRate

    



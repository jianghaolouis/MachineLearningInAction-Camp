import numpy as np
import operator                          #for sorting
import matplotlib.pyplot as plt
import os
import kNN

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
    --------
    feature vector: array.
        the feature vector with the shape (n_features,) 
    '''
    with open(filepath,'r') as dossier:
        raw_image = [row.strip() for row in dossier.readlines()]
        img_feature = np.array(list(''.join(raw_image)),dtype = 'i4')     # i4 means int32
    return img_feature

def images_data(folder):
    '''
    get all digit images into a numpy array which serves as the feature array of kNN and it's labels.

    Parameters:
    -----------
    folder path : str or working path dir
        the name of folder where the images are stored.

    Return:
    -------
    feature array : array
        feature array with the shape (n_sample , n_features). each sample is from a digit and each feature is a pixel.
    label vector : array (n_simple,)
        class labels with the shape(n_sample,).
    '''
    fld = os.listdir(folder)    #this method returns a list contaning the names of the entities in the given directory 
    #initialize feature array (n_sample,n_feature)
    n_s = len(fld)
    n_f = len(load_image(os.path.join(folder,fld[0])))
    f_array = np.empty(n_s, n_f, dtype = 'i4')
    label = np.empty(n_s, dtype = 'i4')

    #put the image vector into each line of feature array
    for i ,f_name in enumerate(fld):
        digitPath = os.path.join(folder,f_name) 
        f_array[i,:] = load_image(digitPath)
        label[i] = f_name.split('_')[0]
    
    return f_array, label

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
        Predict_rs = kNN.kNN_Classify0(x_test[i], _X, _y, k)
        # print( "the classifer came back with : %s, the real answer is :%s" %(Predict_rs,y_test[i]))
        if (Predict_rs != y_test[i]): err_Count += 1.0
    errorRate = err_Count/float(len(x_test))        
    #print("the total error rate is : %f " % errorRate)
    return errorRate
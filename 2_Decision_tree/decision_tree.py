#!/usr/bin/env python
# -*-coding: utf-8 -*-

'''
Decision tree provide knowledge learned from the data. 
Here, the function trains a tree structure using dataset and predicts unknown set.
@jianghaolouis
'''

from math import log
import operator

class dt(object):
    
    def __init__(self):
        pass

    def get_dataset(self,dataset):
        self.dataset = dataset

    def tree_generate(self, Dataset):
        '''
        tree construction function

        Parameters:
        -----------
        Dataset: input dataset of (n_sample,n_feature)
        a: attrubute

        Return:
        -------
        tree structure: 

        '''
        self.dataset = Dataset



    def predict(self, inX):
        pass

    def ft_eval(self, dataset):

        '''
        choose the best attribute that reduce the most of entropy

        Parameters:
        -----------
        dataset： array (n_sample, n_attribute)

        Results:
        -------
        i : int
            the indice of the best feature
        '''
        #for each attribute,find all the types
        n_attri = len(dataset[0]) - 1
        baseline = self.ent(dataset)
        best_gain = 0.0; best_ft = -1
        for i in range(n_attri):
            # take the all the values of attribute i
            value_f = [sample[i] for sample in dataset]
            #an unique set for types
            types = set(value_f)
            
            new_entropy = 0.0
            for value in types:
                sub_dt = self.split(dataset, i, value)
                prob = len(sub_dt)/float(len(dataset))
                new_entropy += prob * self.ent(sub_dt)
            #baseline 
            _gain = baseline - new_entropy
            if _gain > best_gain:
                best_gain = _gain
                best_ft = i
        return best_ft
            







        # calculate the baseline and calculate the info_gain.

    def ent(self,dataset):
        '''
        calculate the entory of dataset. the entropy represents the label purity in data.

        Parameters:
        -----------
        data : feature array of shape (n_sample, n_feature)
        label : label vector of shape (n_sample)
        
        Result:
        -------
        entropy : float.
            entropy metric of the input set.
        '''
        n_sample = len(dataset)
        label_count = {} 
        # create dict for all possible classes
        for instance in dataset:
            lb = instance[-1]                               #the label is the final column
            label_count[lb] = label_count.get(lb, 0) + 1    # if key not existed, create a new one and return 0.

        # sum and log 
        entropy = 0.0
        for key in label_count:
            prob = label_count[key]/float(n_sample)
            entropy -= prob * log(prob,2)
        return entropy
   
    def voting(self,labels):
        '''
        majority voting :find out the most frequent label in the dataset for remarking the dataset class

        Parameters:
        -----------
        labels list : array of shape (n_sample)
        
        Results:
        --------
        labels : int or str
            the most frequent labels in the dataset
        '''
        count = {}
        for _instance in labels:
            _key = _instance[-1] 
            count[_key] = count.get(_key,0) + 1
        st_Count = sorted(count.items(), key = operator.itemgetter(1), reverse = True)
        return st_Count[0][0]

    def split(self,dataset,col,value):
        '''
        split the original dataset and searching in the choosen feature(colomn) for the attribute value 

        Parameters:
        -----------
        dataset : array (n_sample, n_feature + label)
        feature col : int
            the indice of choosen feature for splitting the dataset
        value : int or str
            the feature value who is searched in the feature col

        Results:
        --------
        new dataset : (i_sample, n_feature -1)
            this dataset contains only the samples who had searched value in the feature colomn
            and the new dataset cut out the choosen feature to avoid conflit.
        '''
        new_dt = []
        for instance in dataset: 
            if instance[col] == value:
                #cut out the feature split on 
                sample_v = instance[:col]
                sample_v.extend(instance[col+1:])
                #add to the new dataset
                new_dt.append(sample_v)
        return new_dt






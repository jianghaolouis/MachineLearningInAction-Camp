#!/usr/bin/python3
# -*-coding: utf-8 -*-

'''
Decision tree provide knowledge learned from the data. 
Here, the function trains a tree structure using dataset and predicts unknown set.
@jianghaolouis
'''
from math import log
import operator

class decision_tree(object):
    
    def __init__(self,dataset,attribute):
        self.dataset =dataset
        self.attribute = attribute

    def new_tree(self,_dataset, _a):
        '''
        tree construction function

        Parameters:
        -----------
        Dataset: input dataset of (n_sample,n_feature)
        a: attrubute vector

        Return:
        -------
        tree structure: nesting dict structure
        '''
        _label =[instance[-1] for instance in _dataset]  # return the label list
        # if all equals:
        if _label.count(_label[0])==len(_label):
            return _label[0]
        #if no more feature:
        if len(_dataset[0]) == 1:  
            return self.voting(_label)

        #first split
        _best_ft = self.ft_eval(_dataset)
        _best_a = _a[_best_ft]
        tree = {_best_a:{}}
        #update the attribute list
        #Dont modify _a, because in python the variable is stored by reference. 
        #if modify _a, attribute list will change other where!!!!!
        _b = _a[:]
        _b.remove(_best_a)
        # get the values in ft_i for splitting
        ft_values = set([instance[_best_ft] for instance in _dataset])
        for value in ft_values:
            _sub_a = _b[:]
            tree[_best_a][value] = self.new_tree(self.split(_dataset,_best_ft,value),_sub_a)
        return tree

    def set_tree(self,_tree1):
        '''
        _tree1 : dict format
        '''
        self.tree = _tree1

    def predict(self,_tree,inX,attri_list):
        '''
        inX : feature vector of test sample
        '''
        #return the root key
        f_root = list(_tree.keys())[0]
        #return the value of first key
        f_value = _tree[f_root]
        #return the index (colomn) where to find this attribute in the data 
        a_index = attri_list.index(f_root)
        #feature value in test vector 
        f_x_a = inX[a_index]
        # get the value division for feature f_x_a
        next_value = f_value[f_x_a]
        if isinstance(next_value,dict):
            _label = self.predict(next_value,inX,attri_list)
        else: _label = next_value
        return _label

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
        #for each attribute,find all the values
        n_attri = len(dataset[0]) - 1
        baseline = self.ent(dataset)
        best_gain = 0.0; best_ft = -1
        for i in range(n_attri):
            # take the all the values of attribute i
            value_f = [sample[i] for sample in dataset]
            #an unique set for values
            values = set(value_f)
            
            new_entropy = 0.0
            for value in values:
                sub_dt = self.split(dataset, i, value)
                prob = len(sub_dt)/float(len(dataset))
                new_entropy += prob * self.ent(sub_dt)
            #baseline 
            _gain = baseline - new_entropy
            if _gain > best_gain:
                best_gain = _gain
                best_ft = i
        return best_ft

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
        for _key in labels:
            count[_key] = count.get(_key,0) + 1
        st_Count = sorted(count.items(), key = operator.itemgetter(1), reverse = True)
        return st_Count[0][0]
    
    def store_tree(self,_treeName = 'new_tree.txt'):
        '''
        _path : the folder to store file
        _treeName : 
        '''
        import pickle
        fw = open(_treeName,'wb')
        pickle.dump(self.tree,fw)
        fw.close()

    def grab_tree(self,filename):
        import pickle
        with open(filename,'rb') as fr:
            self.tree = pickle.load(fr)




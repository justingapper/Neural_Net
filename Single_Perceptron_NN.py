# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 11:21:42 2016

@author: jgapper
"""

import numpy as np
 
class Neural_Network(object):
    def __init__(self, lam = .01, steps = 75):
        self.lam = lam
        self.steps = steps
 
    def dot_input(self, X):
        #calc dot product
        return np.dot(X, self.weights)
    
    def logit(self, X):
        #calc logistic function
        return 1/(1+np.exp(-X))
    
    def activation_fxn(self, X):
        #apply activation function
        return self.logit(self.dot_input(X))
    
    def predict(self, X):
        #predict values
        print(np.where(self.activation_fxn(X) >= 0.6, 1, -1))
        
    def train(self, X, y):
        #initiate weights randomly and train perceptron
        self.weights = np.random.random(X.shape[1])
        self.cost = []
                
        for i in range(self.steps):
            output = self.dot_input(X)
            self.error = (y - output)
            self.weights += self.lam * X.T.dot(self.error)
            cost = (self.error**2).sum()/2.0
            print("weights", self.weights)
        return self
        
nn = Neural_Network()
dat = np.array([[0,0],[1,0],[0,1],[1,1]])
dat = np.insert(dat, 0, 1, axis = 1)
y = np.array([0,1,1,1])
nn.train(dat, y)
nn.predict(dat)
 
nn = Neural_Network()
dat = np.array([[0,0],[1,0],[0,1],[1,1]])
dat = np.insert(dat, 0, 1, axis = 1)
y = np.array([0,0,0,1])
nn.train(dat, y)
nn.predict(dat)
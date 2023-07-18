from .utils import *

class TreeGrower:
    '''
    Fits, predicts and plots Decision Trees

    methods
    -------
    fit
    predict
    plot
    
    '''
    def __init__(self,task,cost):
        self.task = task
        self.cost = cost
    def fit(self,X,y,**kwargs):
        *self.model,self.infos = grow_tree(X,y,task=self.task,cost=self.cost,**kwargs)
    def predict(self,X):
        return predict(X,*self.model)
    def plot(self):
        daw_tree(self.model[0])
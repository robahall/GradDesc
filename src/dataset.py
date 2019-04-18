import numpy as np
from sklearn.datasets import make_regression, make_classification, make_blobs

class Data:
    """Generates a data object that can be used to create regression, classification and clustering datasets"""


    #Initializer of attributes
    def __init__(self):
        self.X = 0
        self.Y = 0
        self.coef = 0

    #redundant?
    def inspect_X(self):
        return self.X

    def inspect_Coef(self):
        return self.coef

    def inspect_Y(self):
        return self.Y

class Regression(Data):






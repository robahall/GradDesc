import numpy as np
from sklearn.datasets import make_regression, make_classification, make_blobs

class Data:
    """Generates a data object that can be used to create regression, classification and clustering datasets"""

    def __init__(self, samples, features, informative):
        self.X =
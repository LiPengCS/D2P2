import warnings
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
warnings.filterwarnings("ignore")

class NonlinearTransformer():
    """Normalize the dataset
    Available methods:
    """

    def __init__(self, method='NQT'):
        self.method = method

        if self.method == "NQT":
            self.tf = QuantileTransformer(output_distribution='normal')
        elif self.method == "UQT":
            self.tf = QuantileTransformer(output_distribution='uniform')
        elif self.method == "Power":
            self.tf = PowerTransformer(standardize=False)
        else:
            raise Exception("Invalid normalization method: {}".format(method))
    
    def fit(self, X):
        self.tf.fit(X)

    def transform(self, X):
        X = self.tf.transform(X)
        return X

    def fit_transform(self, X):
        self.fit(X)
        X_trans = self.transform(X)
        return X_trans
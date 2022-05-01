import numpy as np
import pandas as pd

class FeatureSelector(object):
    """Select features for training data.
    Implementation: replace all values by the mean value of each column
    """
    def __init__(self):
        self.method = "drop_feature"
        pass

    def fit(self, X):
        self.mean = X.mean(axis=0)

    def transform(self, X):
        X_trans = np.ones_like(X) * self.mean
        return X_trans

    def fit_transform(self, X):
        self.fit(X)
        X_trans = self.transform(X)
        return X_trans

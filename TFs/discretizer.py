import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

class Discretizer(object):
    """Discretize data

    Args:
        n_bins: number of bins
        strategy: {'uniform', 'quantile', 'kmeans'}
    """
    def __init__(self, n_bins=5, strategy='uniform'):
        self.method = "{}_{}".format(strategy, n_bins)
        self.tf = KBinsDiscretizer(n_bins=n_bins, strategy=strategy, encode='ordinal')

    def fit(self, X):
        self.tf.fit(X)

    def transform(self, X):
        return self.tf.transform(X)

    def fit_transform(self, X):
        return self.tf.fit_transform(X)

import warnings
import time
import numpy as np
from copy import deepcopy
from sklearn.impute import SimpleImputer

class ZSOutlierDetector(object):
    """ Values out of nstd x std are considered as outliers"""
    def __init__(self, nstd=3):
        super(ZSOutlierDetector, self).__init__()
        self.nstd = nstd
        
    def fit(self, X):
        self.lower = []
        self.upper = []
        for i in range(X.shape[1]):
            mean = X[:, i].mean()
            std = X[:, i].std()
            cut_off = std * self.nstd
            l = mean - cut_off
            u = mean + cut_off
            self.lower.append(l)
            self.upper.append(u)
        self.lower = np.array(self.lower).reshape(1, -1)
        self.upper = np.array(self.upper).reshape(1, -1)

    def detect(self, X):
        great = X > self.upper
        low = X < self.lower
        preds = np.logical_or(great, low)
        return preds

class IQROutlierDetector(object):
    """Interquartile Range Methods"""
    def __init__(self, k=1.5):
        super(IQROutlierDetector, self).__init__()
        self.k = k
        
    def fit(self, X):
        self.lower = []
        self.upper = []
        for i in range(X.shape[1]):
            q25 = np.percentile(X[:, i], 25)
            q75 = np.percentile(X[:, i], 75)
            iqr = q75 - q25
            cut_off = iqr * self.k
            l = q25 - cut_off
            u = q75 + cut_off
            self.lower.append(l)
            self.upper.append(u)
        self.lower = np.array(self.lower).reshape(1, -1)
        self.upper = np.array(self.upper).reshape(1, -1)

    def detect(self, X):
        great = X > self.upper
        low = X < self.lower
        preds = np.logical_or(great, low)
        return preds

class MADOutlierDetector(object):
    def __init__(self, nmad=2.5):
        super(MADOutlierDetector, self).__init__()
        self.nmad = nmad
        
    def fit(self, X):
        self.lower = []
        self.upper = []
        for i in range(X.shape[1]):
            median = np.median(X[:, i])
            mad = np.median(np.abs(X[:, i] - median))
            l = median - self.nmad * mad
            u = median + self.nmad * mad
            self.lower.append(l)
            self.upper.append(u)
        self.lower = np.array(self.lower).reshape(1, -1)
        self.upper = np.array(self.upper).reshape(1, -1)

    def detect(self, X):
        great = X > self.upper
        low = X < self.lower
        preds = np.logical_or(great, low)
        return preds

class OutlierCleaner(object):
    """ Detect outliers and repair with mean imputation
    Available methods:
        - 'ZS': detects outliers using the robust Zscore as a function
        - of median and median absolute deviation (MAD)
        - 'IQR': detects outliers using Q1 and Q3 +/- 1.5*InterQuartile Range
        - 'MAD': median absolute deviation
        - 'LOF': detects outliers using Local Outlier Factor
        - 'OCSVM': detects outliers using one-class svm
    """
    def __init__(self, method):
        super(OutlierCleaner, self).__init__()
        self.method = method
        if self.method == "ZS":
            self.detector = ZSOutlierDetector()
        elif self.method == "IQR":
            self.detector = IQROutlierDetector()
        elif self.method == "MAD":
            self.detector = MADOutlierDetector()
        else:
            raise Exception("Invalid normalization method: {}".format(method))

        self.repairer = SimpleImputer()
    
    def fit(self, X):
        self.detector.fit(X)
        indicator = self.detector.detect(X)
        X_clean = deepcopy(X)
        X_clean[indicator] = np.nan
        self.repairer.fit(X_clean)

    def transform(self, X):
        indicator = self.detector.detect(X)
        X_trans = deepcopy(X)
        X_trans[indicator] = np.nan
        X_trans = self.repairer.transform(X_trans)
        return X_trans

    def fit_transform(self, X):
        self.fit(X)
        X_trans = self.transform(X)
        return X_trans

# X = np.zeros((20, 3))
# X[0, :] = 1
# outlier = OutlierCleaner("ZS")
# X_trans = outlier.fit_transform(X)
# print(X)
# print(X_trans)


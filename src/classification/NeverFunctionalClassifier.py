import numpy as np
from sklearn.base import BaseEstimator


class NeverFunctionalClassifier(BaseEstimator):
    def fit(self, X, y=None):
        pass

    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)
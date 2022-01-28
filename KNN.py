import numpy as np
import matplotlib as plt

euclidean = lambda x1, x2: np.sqrt(np.sum((x1 - x2)**2, axis=-1))
manhattan = lambda x1, x2: np.sum(np.abs(x1 - x2), axis=-1)

class KNN:
    def __init__(self, K, dist_fn=euclidean):
        self.K = K
        self.dist_fn = dist_fn

    def fit(self):
        pass

    def predict(self):
        pass

    def evaluate_acc(self):
        pass
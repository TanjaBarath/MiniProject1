import numpy as np
import matplotlib as plt

# real-valued feature vectors
euclidean = lambda x1, x2: np.sqrt(np.sum((x1 - x2)**2, axis=-1))
manhattan = lambda x1, x2: np.sum(np.abs(x1 - x2), axis=-1)
chebyshev = lambda x1, x2: np.max(np.abs(x1 - x2))
minkowski = lambda x1, x2, p: np.sum(np.abs(x1 - x2)**p, axis=-1)**(1/p)
cosine = lambda x1, x2: (np.transpose(x1) * x2)/(np.linalg.norm(x1) * np.linalg.norm(x2))
# discrete feature vectors
hamming = lambda x1, x2: np.count_nonzero(x1 != x2)

class KNN:
    def __init__(self, K, dist_fn=euclidean):
        self.K = K
        self.dist_fn = dist_fn

    def fit(self, x, y):
        # only need training set since KNN is a lazy learner
        self.x = x
        self.y = y
        # each dataset has a 2-valued class
        self.C = 2

    # predict the Class of some test data
    def predict(self, x_test):
        # how many test data points are we predicting
        num_test = x_test.shape[0]
        # get the distance between our trained data and test data
        dist = self.dist_fn(self.x, x_test)
        # ith-row of knns stores the indices of k closest training samples to the ith-test sample
        knns = np.zeros((num_test, self.K), dtype=int)
        # ith-row of y_prob has the probability distribution over C classes
        y_prob = np.zeros((num_test, self.C))
        # sort the neighbors to get the K nearest and get the number of instances of each class
        for i in range(num_test):
            knns[i, :] = np.argsort(dist[i])[:self.K]
            y_prob[i, :] = np.bincount(self.y[knns[i, :]], minlength=self.C)
        # divide by K to get a probability distribution
        y_prob /= self.K
        return y_prob, knns

    def evaluate_acc(self, y_test, y_real):
        pass
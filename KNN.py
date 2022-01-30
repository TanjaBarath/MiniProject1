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
        # gp through every test point and calculate distance with every training point
        distances = []
        for k in range(x_test.shape[0]):
            temp_dist = []
            for i in range(self.x.shape[0]):
                dist = self.dist_fn(x_test.iloc[k,0], self.x.iloc[i,0])
                temp_dist.append(dist)
            distances.append(temp_dist)
        # ith-row of knns stores the indices of k closest training samples to the ith-test sample
        knns = np.zeros((num_test, self.K), dtype=int)
        # ith-row of y_prob has the probability distribution over C classes
        y_prob = np.zeros((num_test, self.C))
        # sort the neighbors to get the K nearest and get the number of instances of each class
        for i in range(num_test):
            knns[i, :] = np.argsort(distances[i])[:self.K]
            temp = np.bincount(self.y.iloc[knns[i, :]]['Class'].to_numpy(), minlength=self.C)
            # bin count gets count from 0 to 2, eliminate the 0 count since it's not a class
            # maybe consider changing class to 0 and 1 rather than 1 and 2?
            y_prob[i, :] = [temp[1], temp[2]]
        # divide by K to get a probability distribution
        y_prob /= self.K
        return y_prob, knns

    def evaluate_acc(self, y_pred, y_test):
        accuracy = np.sum(y_pred == y_test) / y_test.shape[0]
        return accuracy
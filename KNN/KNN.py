# -*- coding: utf-8 -*-
import numpy as np
from collections import Counter

class KNN(object):
    def __init__(self):
        pass
    def train(self, X, y):
        self.X_train = X
        self.y_train = y
    def compute_distance(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            dists[i,:]= np.sqrt(np.sum(np.square(self.X_train - X[i,:]), axis=1))
        return dists
    def predict_labels(self, dists, k=1):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in xrange(num_test):
            labels = []
            sorted_index = np.argsort(dists[i,:])
            for j in range(len(sorted_index)):
                index = sorted_index[j]
                labels.append(self.y_train[index])
            print labels
            nearest_y = labels[0:k]
            c = Counter(nearest_y)
            y_pred[i] = c.most_common(1)[0][0]
        return y_pred
if __name__ == '__main__':
    classifier = KNN()
    X_train = np.array([[1.0, 0.9],[1.0, 1.0], [0.1, 0.2], [0.0, 0.1]])
    y_train = [1, 1, 2, 2]
    X_test = np.array([[1.2, 1.0], [0.1, 0.3]])
    classifier.train(X_train, y_train)
    dists = classifier.compute_distance(X_test)
    print dists
    y_test_pred = classifier.predict_labels(dists)
    print y_test_pred
    
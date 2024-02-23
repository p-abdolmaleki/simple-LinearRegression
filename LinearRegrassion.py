import numpy as np


class LinearRegression:
    def __init__(self, learning_rate, n_iteration):
        self.learning_rate = learning_rate
        self.n_iteration = n_iteration
        self.weights = None
        self.intercept = None
        self.flag = False
    def fit(self, X_train, y_train):
        try:
            n_rows, n_columns = X_train.shape
            weights = np.zeros(n_columns)
            b = 0
            for _ in range(self.n_iteration):
                y_pred = np.dot(X_train, weights) + b
                d_m = (-self.learning_rate / n_columns) * np.dot(X_train.T, y_pred - y_train)
                d_b = (-self.learning_rate / n_columns) * np.sum(y_pred - y_train)
                weights += d_m
                b += d_b
            self.weights = weights
            self.intercept = b
        except:
            n = len(X_train)
            m = 0
            b = 0
            for _ in range(self.n_iteration):
                y_pred = m * X_train + b
                d_m = (-self.learning_rate / n) * np.sum((y_pred - y_train) * X_train)
                d_b = (-self.learning_rate / n) * np.sum(y_pred - y_train)
                m += d_m
                b += d_b
            self.weights = m
            self.intercept = b
            self.flag = True
            
    def predict(self, X_test):
        return np.dot(X_test, self.weights) + self.intercept

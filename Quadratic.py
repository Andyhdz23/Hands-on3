import numpy as np

class QuadraticRegression:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.beta = None
        self.y_pred = None
        self.r_squared = None
        self.fit()

    def fit(self):
        n = self.X.shape[0]
        ones = np.ones((n, 1))
        X_ = np.concatenate((ones, self.X, self.X**2), axis=1)
        self.beta = np.linalg.inv(X_.T @ X_) @ X_.T @ self.y
        self.y_pred = X_ @ self.beta
        ssr = np.sum((self.y_pred - self.y.mean())**2)
        sst = np.sum((self.y - self.y.mean())**2)
        self.r_squared = ssr / sst

    def predict(self, x):
        ones = np.ones((1, 1))
        X_ = np.concatenate((ones, x, x**2), axis=1)
        y_pred = X_ @ self.beta
        return y_pred[0]

    def get_equation(self):
        b0 = self.beta[0]
        b1 = self.beta[1]
        b2 = self.beta[2]
        return f"y = {b0:.2f} + {b1:.2f}x1 + {b2:.2f}x1^2"

    def get_r_squared(self):
        return f"R-squared = {self.r_squared:.2f}"

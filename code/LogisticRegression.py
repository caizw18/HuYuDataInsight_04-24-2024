import numpy as np
import sys

"""This script implements a two-class logistic regression model.
"""

class logistic_regression(object):
	
    def __init__(self, learning_rate, max_iter):
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def fit_BGD(self, X, y, b_size):
        """Train perceptron model on data (X,y) with Batch Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """

		### YOUR CODE HERE
        n_samples, n_features = X.shape
        iter = 0
        self.W = np.zeros((n_features,))
        while iter < self.max_iter:
            iter = iter + 1
            N = n_samples
            S = b_size
            for i in range(0, N, S):
                if i + S > N:
                    k = N - i
                else:
                    k = S
                X_k = X[i:i + k]
                y_k = y[i:i + k]
                Gradient = [self._gradient(p, q) for p, q in zip(X_k, y_k)]
                self.W += self.learning_rate * (-np.mean(Gradient, 0))
		### END YOUR CODE
        return self

    def fit_miniBGD(self, X, y, batch_size):
        """Train perceptron model on data (X,y) with mini-Batch Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.
            batch_size: An integer.

        Returns:
            self: Returns an instance of self.
        """
		### YOUR CODE HERE
        n_samples, n_features = X.shape

        iter = 0
        self.W = np.zeros((n_features,))
        while iter < self.max_iter:
            iter = iter + 1
            N = n_samples
            S = batch_size
            for i in range(0, N, S):
                if i + S > N:
                    k = N - i
                else:
                    k = S
                X_k = X[i:i + k]
                y_k = y[i:i + k]
                Gradient = [self._gradient(p, q) for p, q in zip(X_k, y_k)]
                self.W += -self.learning_rate * np.sum(Gradient)

		### END YOUR CODE
        return self

    def fit_SGD(self, X, y):
        """Train perceptron model on data (X,y) with Stochastic Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
		### YOUR CODE HERE
        n_samples, n_features = X.shape
        iter = 0
        self.W = np.zeros((n_features,))
        while iter < self.max_iter:
            iter = iter + 1
            N = n_samples
            i = np.random.randint(0, N)
            X_i = X[i]
            y_i = y[i]
            Gradient = (-1) * self.learning_rate * self._gradient(X_i, y_i)
            self.W += Gradient
		### END YOUR CODE
        return self

    def _gradient(self, _x, _y):
        """Compute the gradient of cross-entropy with respect to self.W
        for one training sample (_x, _y). This function is used in fit_*.

        Args:
            _x: An array of shape [n_features,].
            _y: An integer. 1 or -1.

        Returns:
            _g: An array of shape [n_features,]. The gradient of
                cross-entropy with respect to self.W.
        """
		### YOUR CODE HERE
        grad = -_x*_y/(np.exp(_y*np.dot(self.W,_x)) + 1)
        return grad
		### END YOUR CODE

    def get_params(self):
        """Get parameters for this perceptron model.

        Returns:
            W: An array of shape [n_features,].
        """
        if self.W is None:
            print("Run fit first!")
            sys.exit(-1)
        return self.W

    def predict_proba(self, X):
        """Predict class probabilities for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds_proba: An array of shape [n_samples, 2].
                Only contains floats between [0,1].
        """
		### YOUR CODE HERE
        probs = []
        for x in X:
            prob = 1 / (np.exp(np.dot(self.w, x)) + 1)
            probs.append(prob)
        probs2 = 1 - probs
        return np.stack((probs, probs2), axis=-1)
		### END YOUR CODE


    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds: An array of shape [n_samples,]. Only contains 1 or -1.
        """
		### YOUR CODE HERE
        # n_samples, n_features = X.shape
        predict = []
        for x in X:
            if np.dot(self.W, x) < 0:
                predict.append(-1)
            else:
                predict.append(1)
        return predict
		### END YOUR CODE

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            score: An float. Mean accuracy of self.predict(X) wrt. y.
        """
		### YOUR CODE HERE
        n_samples, n_features = X.shape
        predicts = self.predict(X)
        return 100 * np.sum(y == predicts) / n_samples
		### END YOUR CODE
    
    def assign_weights(self, weights):
        self.W = weights
        return self


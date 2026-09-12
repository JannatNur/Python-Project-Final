"""
Machine Learning Models Module (DW-7, DW-8, FR-3)
-------------------------------------------------
This module contains transparent, junior-level implementations of:
1. Baseline Models (Mean Regressor & Majority Class Classifier)
2. Linear Regression (Ordinary Least Squares with L2 Ridge stability)
3. Logistic Regression (Sigmoid activation trained via Gradient Descent)
4. K-Nearest Neighbors (KNN Regressor & Classifier with Euclidean distance)
5. Decision Trees (Regression with MSE reduction, Classification with Gini)

Each model has .fit(X, y) and .predict(X) methods, fully explainable for viva.
"""

import math
import numpy as np


# ==========================================
# 1. BASELINE MODELS (DW-7)
# ==========================================

class BaselineMeanRegressor:
    """
    A simple baseline regression model that always predicts the MEAN
    of the training target values (DW-7).
    """
    def __init__(self):
        self.mean_value_ = 0.0
        
    def fit(self, X, y):
        # Calculate mean using Python built-in sum() and len()
        self.mean_value_ = sum(y) / len(y)
        return self
        
    def predict(self, X):
        # Return a constant array containing the mean for every row
        n_samples = len(X)
        return np.full(n_samples, self.mean_value_)


class BaselineMajorityClassifier:
    """
    A simple baseline classification model that always predicts the
    MAJORITY CLASS from the training set (DW-7).
    """
    def __init__(self):
        self.majority_class_ = 0
        
    def fit(self, X, y):
        # Count frequency of each class
        counts = {}
        for label in y:
            counts[label] = counts.get(label, 0) + 1
            
        # Find label with highest count
        best_label = None
        max_count = -1
        for label, count in counts.items():
            if count > max_count:
                max_count = count
                best_label = label
                
        self.majority_class_ = best_label
        return self
        
    def predict(self, X):
        n_samples = len(X)
        return np.full(n_samples, self.majority_class_)
        
    def predict_proba(self, X):
        n_samples = len(X)
        # Returns probability 1.0 for majority class
        return np.full(n_samples, 1.0 if self.majority_class_ == 1 else 0.0)


# ==========================================
# 2. LINEAR REGRESSION (DW-8)
# ==========================================

class LinearRegressionModel:
    """
    Linear Regression using the Normal Equation with Ridge regularization for stability.
    Equation: w = (X^T X + lambda * I)^(-1) X^T y
    Prediction: y_pred = X_test @ w
    """
    def __init__(self, lambda_reg=1e-4):
        self.lambda_reg = lambda_reg
        self.weights_ = None
        self.intercept_ = 0.0
        
    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        
        # Add bias term (column of 1s)
        n_samples, n_features = X.shape
        X_bias = np.c_[np.ones(n_samples), X]
        
        # Regularization matrix (do not regularize intercept)
        I = np.eye(n_features + 1)
        I[0, 0] = 0.0
        
        # Normal equation: w = inv(X_bias.T @ X_bias + lambda * I) @ X_bias.T @ y
        A = X_bias.T @ X_bias + self.lambda_reg * I
        b = X_bias.T @ y
        
        # Solve linear system
        weights_all = np.linalg.solve(A, b)
        self.intercept_ = weights_all[0]
        self.weights_ = weights_all[1:]
        return self
        
    def predict(self, X):
        X = np.array(X, dtype=float)
        return X @ self.weights_ + self.intercept_


# ==========================================
# 3. LOGISTIC REGRESSION (DW-8)
# ==========================================

class LogisticRegressionModel:
    """
    Logistic Regression binary classifier trained via Gradient Descent.
    Sigmoid function: sigma(z) = 1 / (1 + exp(-z))
    Cost function: Binary Cross Entropy Loss
    """
    def __init__(self, learning_rate=0.1, n_iterations=1000, lambda_reg=1e-4):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.lambda_reg = lambda_reg
        self.weights_ = None
        self.bias_ = 0.0
        self.loss_history_ = []
        
    def _sigmoid(self, z):
        # Clip z to avoid overflow in exp(-z)
        z_clipped = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z_clipped))
        
    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        n_samples, n_features = X.shape
        
        # Initialize parameters to zeros
        self.weights_ = np.zeros(n_features)
        self.bias_ = 0.0
        self.loss_history_ = []
        
        # Gradient Descent optimization loop
        for epoch in range(self.n_iterations):
            # 1. Linear combination: z = X @ w + b
            linear_pred = X @ self.weights_ + self.bias_
            
            # 2. Probability prediction through Sigmoid
            y_pred = self._sigmoid(linear_pred)
            
            # 3. Compute gradients
            error = y_pred - y
            dw = (1.0 / n_samples) * (X.T @ error) + (self.lambda_reg / n_samples) * self.weights_
            db = (1.0 / n_samples) * np.sum(error)
            
            # 4. Update parameters
            self.weights_ = self.weights_ - self.lr * dw
            self.bias_ = self.bias_ - self.lr * db
            
            # 5. Compute loss for tracking
            if epoch % 100 == 0 or epoch == self.n_iterations - 1:
                eps = 1e-15
                loss = - (1.0 / n_samples) * np.sum(y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps))
                self.loss_history_.append(loss)
                
        return self
        
    def predict_proba(self, X):
        X = np.array(X, dtype=float)
        linear_pred = X @ self.weights_ + self.bias_
        return self._sigmoid(linear_pred)
        
    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)


# ==========================================
# 4. K-NEAREST NEIGHBORS (KNN) (DW-8)
# ==========================================

class KNNRegressor:
    """
    K-Nearest Neighbors Regressor using Euclidean Distance.
    """
    def __init__(self, k=5):
        self.k = k
        self.X_train_ = None
        self.y_train_ = None
        
    def fit(self, X, y):
        self.X_train_ = np.array(X, dtype=float)
        self.y_train_ = np.array(y, dtype=float)
        return self
        
    def _predict_single(self, x):
        # Calculate Euclidean distance: sqrt(sum((x_train - x)^2))
        distances = np.sqrt(np.sum((self.X_train_ - x) ** 2, axis=1))
        # Get indices of k smallest distances
        k_indices = np.argsort(distances)[:self.k]
        # Average the target values of k neighbors
        k_neighbor_values = self.y_train_[k_indices]
        return np.mean(k_neighbor_values)
        
    def predict(self, X):
        X = np.array(X, dtype=float)
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)


class KNNClassifier:
    """
    K-Nearest Neighbors Classifier using Euclidean Distance and Majority Voting.
    """
    def __init__(self, k=5):
        self.k = k
        self.X_train_ = None
        self.y_train_ = None
        
    def fit(self, X, y):
        self.X_train_ = np.array(X, dtype=float)
        self.y_train_ = np.array(y, dtype=int)
        return self
        
    def _predict_single(self, x):
        distances = np.sqrt(np.sum((self.X_train_ - x) ** 2, axis=1))
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train_[k_indices]
        
        # Majority vote
        counts = {}
        for label in k_nearest_labels:
            counts[label] = counts.get(label, 0) + 1
            
        majority_label = max(counts, key=counts.get)
        prob = counts.get(1, 0) / self.k
        return majority_label, prob
        
    def predict(self, X):
        X = np.array(X, dtype=float)
        results = [self._predict_single(x)[0] for x in X]
        return np.array(results)
        
    def predict_proba(self, X):
        X = np.array(X, dtype=float)
        probs = [self._predict_single(x)[1] for x in X]
        return np.array(probs)


# ==========================================
# 5. DECISION TREE (DW-8)
# ==========================================

class _Node:
    """Helper node class for decision tree structure."""
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
    def is_leaf(self):
        return self.value is not None


class DecisionTreeRegressorModel:
    """
    Decision Tree Regressor implementing binary recursive splitting using Variance Reduction.
    """
    def __init__(self, max_depth=4, min_samples_split=5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        
    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        self.root = self._build_tree(X, y, depth=0)
        return self
        
    def _variance(self, y):
        if len(y) <= 1:
            return 0.0
        return np.var(y) * len(y)
        
    def _best_split(self, X, y):
        best_feat, best_thresh = None, None
        best_score = float('inf')
        n_samples, n_features = X.shape
        
        if n_samples < self.min_samples_split:
            return None, None
            
        current_variance = self._variance(y)
        
        for feat_idx in range(n_features):
            values = np.unique(X[:, feat_idx])
            for val in values:
                left_mask = X[:, feat_idx] <= val
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                    
                score = self._variance(y[left_mask]) + self._variance(y[right_mask])
                if score < best_score:
                    best_score = score
                    best_feat = feat_idx
                    best_thresh = val
                    
        return best_feat, best_thresh
        
    def _build_tree(self, X, y, depth):
        n_samples = len(y)
        if depth >= self.max_depth or n_samples < self.min_samples_split or len(np.unique(y)) == 1:
            return _Node(value=np.mean(y))
            
        feat_idx, thresh = self._best_split(X, y)
        if feat_idx is None:
            return _Node(value=np.mean(y))
            
        left_mask = X[:, feat_idx] <= thresh
        right_mask = ~left_mask
        
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        return _Node(feature_idx=feat_idx, threshold=thresh, left=left_child, right=right_child)
        
    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
        
    def predict(self, X):
        X = np.array(X, dtype=float)
        return np.array([self._traverse_tree(x, self.root) for x in X])


class DecisionTreeClassifierModel:
    """
    Decision Tree Classifier implementing binary recursive splitting using Gini Impurity.
    """
    def __init__(self, max_depth=4, min_samples_split=5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        
    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=int)
        self.root = self._build_tree(X, y, depth=0)
        return self
        
    def _gini(self, y):
        if len(y) == 0:
            return 0.0
        p1 = np.sum(y == 1) / len(y)
        p0 = 1.0 - p1
        return 1.0 - (p0 ** 2 + p1 ** 2)
        
    def _best_split(self, X, y):
        best_feat, best_thresh = None, None
        best_gain = -1.0
        n_samples, n_features = X.shape
        
        if n_samples < self.min_samples_split:
            return None, None
            
        parent_gini = self._gini(y)
        
        for feat_idx in range(n_features):
            values = np.unique(X[:, feat_idx])
            for val in values:
                left_mask = X[:, feat_idx] <= val
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                    
                n_left, n_right = np.sum(left_mask), np.sum(right_mask)
                weighted_gini = (n_left / n_samples) * self._gini(y[left_mask]) + (n_right / n_samples) * self._gini(y[right_mask])
                gain = parent_gini - weighted_gini
                
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat_idx
                    best_thresh = val
                    
        return best_feat, best_thresh
        
    def _build_tree(self, X, y, depth):
        n_samples = len(y)
        majority_val = 1 if np.sum(y == 1) >= (n_samples / 2.0) else 0
        
        if depth >= self.max_depth or n_samples < self.min_samples_split or len(np.unique(y)) == 1:
            return _Node(value=majority_val)
            
        feat_idx, thresh = self._best_split(X, y)
        if feat_idx is None:
            return _Node(value=majority_val)
            
        left_mask = X[:, feat_idx] <= thresh
        right_mask = ~left_mask
        
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        return _Node(feature_idx=feat_idx, threshold=thresh, left=left_child, right=right_child)
        
    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
        
    def predict(self, X):
        X = np.array(X, dtype=float)
        return np.array([self._traverse_tree(x, self.root) for x in X])


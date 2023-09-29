import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Logit:
    def __init__(self, batch_size=1, n_iters=100, learning_rate=1):
        """
        Constructor for the Logit class.
        Initializes parameters like learning rate, iterations, and batch size.
        """
        self.lr = learning_rate
        self.n_iters = n_iters
        self.batch_size = batch_size
        self.activation_function = self._sigmoid_func
        self.loss_function = self._log_loss
        self.weights = None
        self.loss = None

    def _min_max_scaling(self, X):
        """
        Scale features between 0 and 1 using Min-Max scaling.
        """
        col_max = np.max(X, axis=0)
        col_min = np.min(X, axis=0)
        return np.divide(X - col_min, col_max - col_min)
    
    def _linear_output(self, X):
        """
        Compute the linear combination of features and weights.
        """
        return np.matmul(X, self.weights)

    def _sigmoid_func(self, x):
        """
        The Sigmoid activation function.
        """
        return 1 / (1 + np.exp(-1 * x))
    
    def _log_loss(self, y, y_pred):
        """
        Compute the log loss for given true labels and predictions.
        """
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    
    def _compute_gradient(self, X, y, y_predicted):
        gradient = np.mean(np.matmul(X.T, (y_predicted - y)))
        return gradient
    
    def predict(self, X):
        """
        Predict the class labels for given features.
        """
        linear_output = self._linear_output(X)
        y_predicted = self.activation_functionc(linear_output)
        return y_predicted
    
    def fit(self, X, y):
        """
        Fit the model to the training data.
        """
        n_samples, n_features = X.shape

        # Initialize weights (including bias)
        self.weights = np.zeros(n_features)
        
        for _ in range(self.n_iters):
            linear_output = self._linear_output(X)
            y_predicted = self.activation_function(linear_output)
            self.loss = self.loss_function(y, y_predicted)

# Read the dataset
df = pd.read_csv(r"D:\ML\Portfolio\Projects\NumPy ML\Logistic Regression\grades_dataset.csv")

# Data preprocessing
df.Passed = df.Passed.replace(to_replace=['yes', 'no'], value=[1, 0])
df = df.sort_values(by='Passed', ignore_index=True)

# Plotting the data
plt.scatter(df[:26].HoursStudied, df[:26].PreviousGrade, label='Failed')
plt.scatter(df[26:].HoursStudied, df[26:].PreviousGrade, label='Passed')
plt.xlabel('Hours Studied')
plt.ylabel('Previous Grade')
plt.legend(loc='best')
#plt.show()

# Splitting the dataset
X = df.drop(labels='Passed', axis=1).to_numpy()
y = df.Passed.to_numpy()

# Append bias term to the input features
n_samples, n_features = X.shape
bias_inputs = np.ones((n_samples, 1))
X = np.concatenate((X, bias_inputs), axis=1)

# Splitting the dataset
seed = 5
np.random.seed(seed)
train_set = np.random.choice(len(X), round(len(X) * 0.6), replace=False)
test_set = np.array(list(set(range(len(X))) - set(train_set)))

train_X = X[train_set]
train_y = y[train_set]
test_X = X[test_set]
test_y = y[test_set]

# Fit and print loss
log_reg = Logit()
log_reg.fit(train_X, train_y)
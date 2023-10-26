import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

class Logit:
    def __init__(self, batch_size=1, n_iters=10, learning_rate=1):
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
        self.lin_output = None

    def _linear_output(self, X):
        """
        Compute the linear combination of features and weights.
        """
        self.lin_output = np.matmul(X, self.weights)
        return self.lin_output
        #return np.matmul(X, self.weights)

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
        gradient = np.matmul(X.T, (y_predicted - y)) / len(y_predicted)
        return gradient
    
    def predict(self, X):
        """
        Predict the class labels for given features.
        """
        linear_output = self._linear_output(X)
        y_predicted = self.activation_function(linear_output)
        return y_predicted
    
    def fit(self, X, y):
        """
        Fit the model to the training data.
        """
        n_samples, n_features = X.shape

        # Initialize weights (including bias)
        self.weights = np.zeros(n_features)
        
        for _ in range(self.n_iters):
            print(_)
            y_predicted = self.predict(X)
            lin_abs = np.abs(self.lin_output)
            print('Linear output:', np.max(lin_abs))
            print('Predictions:', y_predicted)
            loss = self.loss_function(y, y_predicted)
            print('Loss:', loss)
            dw = self._compute_gradient(X, y, y_predicted)
            print('Gradient:', dw)
            self.weights -= self.lr * dw
            print('Updated weights: ', self.weights)

def min_max_scaling(X):
        """
        Scale features between 0 and 1 using Min-Max scaling.
        """
        col_max = np.max(X, axis=0)
        col_min = np.min(X, axis=0)
        return np.divide(X - col_min, col_max - col_min)

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
plt.show()

# Split the dataset labels
X = df.drop(labels='Passed', axis=1).to_numpy()
y = df.Passed.to_numpy()

# Feature scaling
X_scaled = min_max_scaling(X)

# Plotting the scaled data
passed_mask = y == 1
failed_mask = y == 0

# Extracting values for students who passed and failed
X_passed = X_scaled[passed_mask]
X_failed = X_scaled[failed_mask]

# Plotting
plt.figure()
plt.scatter(X_failed[:, 0], X_failed[:, 1], label='Failed', color='red')
plt.scatter(X_passed[:, 0], X_passed[:, 1], label='Passed', color='green')
plt.xlabel('Scaled Hours Studied')
plt.ylabel('Scaled Previous Grade')
plt.legend(loc='best')
plt.title('Scaled Input Data')
plt.show()

# Append bias term to the input features
n_samples, n_features = X.shape
bias_inputs = np.ones((n_samples, 1))
X = np.concatenate((X, bias_inputs), axis=1)

# Split the dataset
seed = 5
np.random.seed(seed)
train_set = np.random.choice(len(X_scaled), round(len(X_scaled) * 0.6), replace=False)
test_set = np.array(list(set(range(len(X_scaled))) - set(train_set)))

train_X = X_scaled[train_set]
train_y = y[train_set]
test_X = X_scaled[test_set]
test_y = y[test_set]

# Fit and print loss
log_reg = Logit()
log_reg.fit(train_X, train_y)
pred = log_reg.predict(train_X)
#print(pred)

'''
# Visualize the loss landscape for the composite function
weight_range = np.linspace(-10, 10, 100)
W1, W2 = np.meshgrid(weight_range, weight_range)

Z = np.zeros(W1.shape)
for i in range(W1.shape[0]):
    for j in range(W1.shape[1]):
        # a. Compute the linear output for the current weight values
        linear_output = np.matmul(train_X[:,:2], np.array([W1[i,j], W2[i,j]]).T)
        # b. Apply the activation function
        predictions = log_reg._sigmoid_func(linear_output)
        # c. Compute the loss
        Z[i,j] = log_reg._log_loss(train_y, predictions)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(W1, W2, Z, cmap='viridis')
ax.set_title('Loss Landscape for Composite Function')
ax.set_xlabel('Weight 1')
ax.set_ylabel('Weight 2')
ax.set_zlabel('Loss')
plt.show()
'''
import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, learning_rate=1, n_iters=1000, tolerance=1e-9):
        """
        Initialize a new instance of the Perceptron class.

        Parameters:
        learning_rate (float): The learning rate to use when adjusting the weights and bias.
        n_iters (int): The maximum number of iterations to use in the training process.
        tolerance (float): The error tolerance for the neuron's linear output.
        """
        self.lr = learning_rate
        self.n_iters = n_iters
        self.tolerance = tolerance
        self.activation_func = self._unit_step_func
        self.weights = None
        self.bias = None
        self.epoch_counter = 0
        self.update_counter = 0

    def fit(self, X, y):
        """
        Train the perceptron model using the provided training data.

        Parameters:
        X (array): The input features for the training data.
        y (array): The target outputs for the training data.
        """
        n_samples, n_features = X.shape

        # Init parameters
        self.weights = np.ones(n_features)
        self.bias = 1

        y_ = np.array([1 if i > 0 else 0 for i in y]) # Converts labels to (0, 1)

        for _ in range(self.n_iters):
            # Keep track if an update was made
            update_made = False

            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                if abs(linear_output) < self.tolerance:
                    linear_output = 0 # Correct for floating point error
                y_predicted = self.activation_func(linear_output)
                
                # Perceptron update rule
                update = self.lr * (y_[idx] - y_predicted) # Residual (Error function)

                if update != 0: # Error function minimisation
                    self.update_counter += 1
                    update_made = True
                    self.weights += update * x_i
                    self.bias += update
                    
            # Training epoch counter       
            if update_made:
                self.epoch_counter += 1
            else:
                break # If no update was made in this epoch, stop the training.

    def predict(self, X):
        """
        Predict the class labels for the provided data points.

        Parameters:
        X (array): The data points to classify.
        """
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

    def _unit_step_func(self, x):
        """
        The unit step function with 0.5 output for zero argument.

        Parameters:
        x (array): Neuron output(s), incl. for batch predictions
        """
        return np.where(x < 0, 0, 
                       np.where(x == 0, 0.5, 1))
    
    def plot_decision_boundary(self, X, y):
        """
        Plots the decision boundary of the perceptron given input (X) and output (y)
        """
        fig, ax = plt.subplots()
        
        # Get min and max values and add some padding
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        
        # Generate a grid of points with distance h between them
        h = 0.02
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot the contour and training examples
        plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)

        # Calculate line parameters
        w1, w2 = self.weights[0], self.weights[1]
        b = self.bias

        # Create line
        x_values = np.linspace(x_min, x_max, 100)
        y_values = - (b + w1*x_values) / w2

        # Plot decision boundary line
        ax.plot(x_values, y_values, "k--", label=f"{w1:.2f}x + {w2:.2f}y = {-b:.2f}")

        # Add plot title
        plt.title('Weight adjustment, step {}'.format(self.update_counter))

        plt.legend()
        plt.show() 


# Linear logic gate inputs 
X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])

# Linear logic gate labels
y_and = np.array([0, 0, 0, 1]) # AND gate labels
y_or = np.array([0, 1, 1, 1]) # OR gate labels

# Collect training evaluation data
learning_rate = []
epochs = []
updates = []

# For AND gate

# Fit model using various learning rates
for lr in np.linspace(0.1, 1.0, num=10):
    p_and = Perceptron(learning_rate=round(lr,1), n_iters=1000) # Includes floating-point error correction for resulting learning rate values
    p_and.fit(X, y_and)
    learning_rate.append(round(lr,1))
    epochs.append(p_and.epoch_counter)
    updates.append(p_and.update_counter)

# Display training related stats
for lr, u, e in zip(learning_rate, updates, epochs):
    print('Learning rate {}: updates taken: {}, epochs taken: {}'.format(lr, u, e))

# Identify lowest update count encountered during training
min_updates = min(updates)

# Iterate over learning_rate, updates, and epochs to obtain a list of best learning rates
print("\nResults for the minimum number of updates:")
best_lr = []
for lr, u, e in zip(learning_rate, updates, epochs):
    if u == min_updates:
        best_lr.append(lr)
        print('Learning rate {}: updates taken: {}, epochs taken: {}'.format(lr, u, e))
        
# Fit the model based on the best learning rate, using max learning rate if more than one learning rate with identical update count
print('\nFitting the model based on the best learning rate: {}'.format(max(best_lr)))
p_and = Perceptron(learning_rate=max(best_lr), n_iters=1000)
p_and.fit(X, y_and)

# Make predictions, plot decision boundary and display final parameters
predictions_and = p_and.predict(X)
print("Training completed: update count: {}, epochs taken: {}".format(p_and.update_counter, p_and.epoch_counter))
print("Final bias is: {}, final weights are: {}".format(round(p_and.bias,2), p_and.weights.round(2)))
print("AND gate prediction:", predictions_and)
p_and.plot_decision_boundary(X, y_and)
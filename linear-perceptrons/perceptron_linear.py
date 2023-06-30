import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, learning_rate=1, n_iters=1000, tolerance=1e-9):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.tolerance = tolerance
        self.activation_func = self._unit_step_func
        self.weights = None
        self.bias = None
        self.epoch_counter = 0
        self.update_counter = 0

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.array([1 if i > 0 else 0 for i in y])

        for _ in range(self.n_iters):

            # Keep track if an update was made
            update_made = False

            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                if abs(linear_output) < self.tolerance:
                    linear_output = 0
                y_predicted = self.activation_func(linear_output)
                print('Sample {}: {}'.format(idx + 1, x_i))
                print('Linear output: {}, predicted class: {}, correct class: {}'.format(linear_output, y_predicted, y[idx]))
                # Perceptron update rule

                if linear_output == 0 and y_[idx] == 0:
                    update = self.lr * (-1)
                else:
                    update = self.lr * (y_[idx] - y_predicted)
                
                #update = self.lr * (y_[idx] - y_predicted)

                if update != 0:
                    self.update_counter += 1
                    print('Update #{}'.format(self.update_counter))
                    print('Bias update: {}, weights update: {}'.format(update, update * x_i))
                    #print(linear_output, y_predicted)
                    update_made = True
                    self.weights += update * x_i
                    self.bias += update
                    print('Bias after update: {}, weights after update: {}'.format(self.bias, self.weights))
                    #print('{}x+{}y={}'.format(self.weights[0], self.weights[1], self.bias * -1))

            # Training epoch counter       
            if update_made:
                self.epoch_counter += 1
                print('Epoch completed: {}'.format(self.epoch_counter))
            else:
                break # If no update was made in this epoch, stop the training

        print("Training completed. Epochs taken: {}, update count: {}".format(self.epoch_counter, self.update_counter))

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

    def _unit_step_func(self, x):
        return np.where(x>0, 1, 0)
    
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

        plt.legend()
        plt.show() 

# For AND gate
X_and = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
y_and = np.array([0, 0, 0, 1])

p_and = Perceptron(learning_rate=1, n_iters=1000)
p_and.fit(X_and, y_and)
predictions_and = p_and.predict(X_and)

#p_and.plot_decision_boundary(X_and, y_and)

print("AND gate prediction:", predictions_and)
'''
# For OR gate
X_or = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
y_or = np.array([-1, 1, 1, 1])

p_or = Perceptron(learning_rate=1, n_iters=1000)
p_or.fit(X_or, y_or)
predictions_or = p_or.predict(X_or)

p_or.plot_decision_boundary(X_or, y_or)

print("OR gate prediction:", predictions_or)
'''
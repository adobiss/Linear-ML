import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.1, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = self._unit_step_func
        self.weights = None
        self.bias = None
        self.epoch_counter = 0

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
                y_predicted = self.activation_func(linear_output)

                # Perceptron update rule
                update = self.lr * (y_[idx] - y_predicted)
                if update != 0:
                    update_made = True
                    self.weights += update * x_i
                    self.bias += update

            # Training epoch counter       
            if update_made:
                self.epoch_counter += 1

            # If no update was made in this epoch, stop the training
            if not update_made:
                break
        print("Training completed. Epochs taken: {}".format(self.epoch_counter))

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

    def _unit_step_func(self, x):
        return np.where(x>0, 1, 0)

# For AND gate
X_and = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
y_and = np.array([0, 0, 0, 1])

p_and = Perceptron(learning_rate=0.1, n_iters=1000)
p_and.fit(X_and, y_and)
predictions_and = p_and.predict(X_and)

print("AND gate prediction:", predictions_and)

# For OR gate
X_or = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
y_or = np.array([-1, 1, 1, 1])

p_or = Perceptron(learning_rate=0.1, n_iters=1000)
p_or.fit(X_or, y_or)
predictions_or = p_or.predict(X_or)
print("OR gate prediction:", predictions_or)
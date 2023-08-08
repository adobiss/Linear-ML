import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Logit:
    def __init__(self, batch_size, n_iters, learning_rate=1):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.batch_size = batch_size
        self.activation_function = self.sigmoid_func
        self.weights = None
        self.bias = None

    def _min_max_scaling(self, X):
        col_max = np.max(X, axis=0)
        col_min = np.min(X, axis=0)
        return np.divide(X - col_min, col_max - col_min)
    
    def _linear_output(self, X):
        linear_output = np.matmul(X, self.weights) + self.bias

    
    
    def fit(self, X):
        n_samples, n_features = train_X.shape




df = pd.read_csv(r"D:\ML\Portfolio\Projects\NumPy ML\Logistic Regression\grades_dataset.csv")
df.head(5)
df.shape
df['Passed'].value_counts()
df.Passed = df.Passed.replace(to_replace=['yes', 'no'], value=[1, 0])
df.head(5)
df = df.sort_values(by='Passed', ignore_index='True')
df.head(5)

plt.scatter(df[:25].HoursStudied, df[:25].PreviousGrade, label='Failed')
plt.scatter(df[25:].HoursStudied, df[25:].PreviousGrade, label='Passed')
plt.xlabel('Hours Studied')
plt.ylabel('Previous Grade')
plt.legend(loc='best')
#plt.show()

X = df.drop(labels='Passed', axis=1).to_numpy()
y = df.Passed.to_numpy()

seed = 5
np.random.seed(seed)
train_set = np.random.choice(len(X), round(len(X) * 0.6), replace=False)
test_set = np.array(list(set(range(len(X))) - set(train_set)))

train_X = X[train_set]
train_y = y[train_set]
test_X = X[test_set]
test_y = y[test_set]


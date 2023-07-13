import numpy as np
import matplotlib.pyplot as plt

def plot_chart(X, w, attempt_counter):
    """
    A function to plot the linear discriminant function which represents a decision boundary
    for a logic gate.

    Parameters:
    X (array): Input matrix
    w (array): Weight vector
    attempt_counter (int): Step count for weight adjustment
    """
    # Extracting x, y coordinates from input matrix
    x_coordinates = X[:, 1]
    y_coordinates = X[:, 2]

    x = np.linspace(-np.max(X), np.max(X))  # Domain of the function for plotting

    w1, w2, w3 = np.split(w, w.shape[0])  # Splitting weight vector into individual coefficients

    if w3.item() == 0:  # Handling vertical lines where slope is undefined
        print('Not a function!')
    else:
        # Calculating y for given x according to weight coefficients
        y = (-w2.item() * x - w1.item()) / w3.item()
        y2 = (-w2.item() * x - w1.item() + 1) / w3.item()
        y3 = (-w2.item() * x - w1.item() - 1) / w3.item()

        a = w2.item()
        b = w3.item()
        c = w1.item()

        # Formatting the coefficients (a, b, c)
        if a - int(a) == 0:
            a = int(a)
        else:
            a = round(a, 2)

        if b - int(b) == 0:
            b = int(b)
        else:
            b = round(b, 2)

        if c - int(c) == 0:
            c = int(c)
        else:
            c = round(c, 2)

        graph_equation = '{0}x {1}y {2}=0'.format(a, b, c)  # Constructing the equation of the line
        
        # Formatting equation of the line
        eq_sign_format = {'0x': '',
                    'x ': 'x+',
                    '0y': '',
                    'y 0': 'y',
                    'y ': 'y+',
                    '+-': '-',
                    #' ': ''
                    }

        def get_value(k): 
            for key, value in eq_sign_format.items(): 
                 if k == key: 
                     return value
        
        for i in eq_sign_format.keys():
            if i in graph_equation:
                graph_equation = graph_equation.replace(i, get_value(i))

        # Plotting the line, adding plot title and labels
        plt.plot(x, y, label=graph_equation, color='blue')
        plt.title('Weight adjustment, step {}'.format(attempt_counter))
        plt.xlabel('x axis')
        plt.ylabel('y axis')
        plt.legend(loc='upper right')
        plt.grid(alpha=.4,linestyle='--')  # Adding grid lines for better readability

        # Colouring the points based on their class
        coordinate_colors = []
        for i in range(Y.shape[0]):
            if Y[i] == -1:
                coordinate_colors.append('Red')
            if Y[i] == 1:
                coordinate_colors.append('Green')

        plt.scatter(x_coordinates, y_coordinates, color=coordinate_colors)  # Plotting the points
        plt.show() 

# Creating a dataset for AND gate
dataset = np.array([
    [0, 0, -1],  # 0 and 0 => -1
    [1, 0, -1],  # 1 and 0 => -1
    [0, 1, -1],  # 0 and 1 => -1
    [1, 1, 1]    # 1 and 1 => 1
])

X_inputs = dataset[:, :-1]
Y = dataset[:, -1:]

# Adding bias inputs to the input array
bias_inputs = np.ones((X_inputs.shape[0], 1), dtype=int)
X = np.concatenate((bias_inputs, X_inputs), axis=1)

X_samples = np.split(X, 4)

w = np.zeros((X.shape[1], 1), dtype=int)  # Initialising the weight vector

attempt_counter = 0

# Training loop using the perceptron update rule
while np.any(np.matmul(X, w) * Y <= 0): # Error function minimisation
    for i in range(X.shape[0]):
        if np.matmul(X_samples[i], w) * Y[i] <= 0: # Error function
            w = np.add(w.T, * X_samples[i] * Y[i])
            w = w.T
            attempt_counter += 1

print("Final weights are {}, {} training steps taken".format(w.flatten().tolist(), attempt_counter))

plot_chart(X, w, attempt_counter)
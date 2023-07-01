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
    
    x_coordinates = X[:, 1]  # Extracting x-coordinates from input matrix
    y_coordinates = X[:, 2]  # Extracting y-coordinates from input matrix

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

        # Plotting the line
        plt.plot(x, y, label=graph_equation, color='blue')
        plt.title('Weight adjustment, step {}'.format(attempt_counter))  # Adding title to the plot
        plt.xlabel('x axis')  # Setting x-axis label
        plt.ylabel('y axis')  # Setting y-axis label
        plt.legend(loc='upper right')  # Adding legend
        plt.grid(alpha=.4,linestyle='--')  # Adding grid lines for better readability

        # Colouring the points based on their class
        coordinate_colors = []
        for i in range(Y.shape[0]):
            if Y[i] == -1:
                coordinate_colors.append('Red')
            if Y[i] == 1:
                coordinate_colors.append('Green')

        plt.scatter(x_coordinates, y_coordinates, color=coordinate_colors)  # Plotting the points
        plt.show()  # Displaying the plot

# Creating a dataset for AND gate
dataset = np.array([
    [0, 0, -1],  # 0 and 0 => -1
    [0, 1, -1],  # 0 and 1 => -1
    [1, 0, -1],  # 1 and 0 => -1
    [1, 1, 1]    # 1 and 1 => 1
])

X_inputs = dataset[:, :-1]  # Extracting input values from the dataset
Y = dataset[:, -1:]  # Extracting output values (classes) from the dataset

bias_inputs = np.ones((X_inputs.shape[0], 1), dtype=int)  # Creating a bias input array of ones
X = np.concatenate((bias_inputs, X_inputs), axis=1)  # Adding bias inputs to the input array

samples = np.split(X, 4)  # Splitting the input matrix into individual vectors

w = np.zeros((X.shape[1], 1), dtype=int)  # Initializing the weight vector with zeros

attempt_counter = 0  # Initializing the counter for weight adjustment steps
lr = 1  # Setting the learning rate

# Training loop using the Perceptron Learning Algorithm
while np.any(np.matmul(X, w) * Y <= 0): # Error function
    for i in range(X.shape[0]):
        if np.matmul(samples[i], w) * Y[i] <= 0: # Error function
            w = np.add(w.T, lr * samples[i] * Y[i])  # Updating weights
            w = w.T
            attempt_counter += 1  # Incrementing step counter
            print(w.T)

plot_chart(X, w, attempt_counter)  # Plotting the results

print("Final weights are {}, {} training steps taken".format(w.flatten().tolist(), attempt_counter))  # Printing the final weights and step count
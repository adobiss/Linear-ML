# Linear Logic Gate v1

This Python script implements a Perceptron algorithm for a simple binary AND logic gate. The script also includes a function to plot the decision boundary for the logic gate. 

## Prerequisites

The script requires the following Python packages:

- numpy
- matplotlib

If not installed, use the package manager [pip](https://pip.pypa.io/en/stable/) to install them.

```bash
pip install numpy matplotlib
```

## Description of the script

The script begins by defining a helper function plot_chart(X, w, attempt_counter) which plots the decision boundary of the logic gate based on the weights from the Perceptron algorithm.

Then, it defines an AND gate dataset, where 0 and 1 represent the input values, and -1 and 1 represent the output values (negative values are used to differentiate between the "off" state and the bias).

Next, the bias inputs are added to the input array, and the weight vector is initialised with zeroes.

After the setup, the script enters the training loop using the perceptron update rule. The training continues until all data points are classified correctly (i.e., until the dot product of the weights and inputs yields a positive output for all data points). The weight vector is updated whenever a data point is misclassified.

Finally, the script outputs the final weights and the number of training steps taken, and calls the plot_chart() function to display the decision boundary of the logic gate.

## Output
The script prints the final weight vector, and the number of training steps taken. It also displays a plot of the decision boundary, with the points coloured according to their class (red for -1, green for 1):

_Final weights are [-4, 2, 3], 18 training steps taken_.

![AND Gate Decision Boundary](https://github.com/adobiss/numpy-ml/assets/95383833/3190f736-333c-4ef0-b6ec-0b2b37895812)

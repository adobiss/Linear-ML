# Linear Logic Gate v2
This Python script applies the Perceptron algorithm to model binary logic gates like AND, OR and includes decision boundary visualisation.

## Overview
The Perceptron in this version uses a unit step activation function and implements the standard update rule. It also contains an error tolerance feature for floating point errors, decision boundary visualisation and a learning example for the AND logic function.

## Prerequisites
The script requires the following Python packages:

- numpy
- matplotlib

If not installed, use the package manager [pip](https://pip.pypa.io/en/stable/) to install them.

```bash
pip install numpy matplotlib
```

## Features
### The Perceptron class
* **\_\_init\_\_ (self, learning_rate=1, n_iters=1000, tolerance=1e-9)**: constructs a new Perceptron instance with specific learning parameters.
* **fit(self, X, y)**: trains the model using given data.
* **predict(self, X)**: predicts class labels for the given data points.
* **_unit_step_func(self, x)**: defines a unit step function with a return value of 0.5 for zero argument.
* **plot_decision_boundary(self, X, y)**: visualises the decision boundary of the perceptron for given input (X) and output (y).

## Description of the script
The script defines a Perceptron class and sets up datasets for AND and OR logic gates using 0 and 1 for input and output values.

It explores learning rates using the AND gate as an example, ranging from 0.1 to 1.0. For each rate, it creates a Perceptron model, trains it on the AND gate data and logs the epoch and weight update count during training.

The training loop itself applies the Perceptron update rule until all data points are correctly classified or maximum iterations reached. The rule adjusts weights and bias when model's prediction differs from true output using **update = learning_rate * (true_output - predicted_output)**.

If no updates are made in an epoch (meaning no misclassifications occurred) all data points are classified correctly and the training is stopped.

Once the optimal learning rate is identified, it fits the model and makes predictions, also plotting the decision boundary.

## Output
The script outputs training statistics, the optimal learning rate, total updates and epochs, final weight vector as well as predictions for each training data input. It also displays a plot of the decision boundary with data points colour-coded by class (red for 0, blue for 1):

Learning rate 0.1: updates taken: 16, epochs taken: 6  
Learning rate 0.2: updates taken: 8, epochs taken: 3  
Learning rate 0.3: updates taken: 7, epochs taken: 2  
Learning rate 0.4: updates taken: 14, epochs taken: 5  
Learning rate 0.5: updates taken: 16, epochs taken: 6  
Learning rate 0.6: updates taken: 12, epochs taken: 5  
Learning rate 0.7: updates taken: 5, epochs taken: 2  
Learning rate 0.8: updates taken: 12, epochs taken: 5  
Learning rate 0.9: updates taken: 12, epochs taken: 5  
Learning rate 1.0: updates taken: 14, epochs taken: 5  

Results for the minimum number of updates:  
Learning rate 0.7: updates taken: 5, epochs taken: 2  

Fitting the model based on the best learning rate: 0.7  
Training completed: update count: 5, epochs taken: 2  
Final bias is: -1.1, final weights are: [0.3 1. ]  
AND gate prediction: [0. 0. 0. 1.]  
![AND Gate Decision Boundary](https://github.com/adobiss/numpy-ml/assets/95383833/8eb9efdc-1a03-4330-8074-ee594dc87c29)

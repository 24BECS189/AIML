import numpy as np


# -------------------------------
# Single Layer Perceptron - AND
# -------------------------------

def step_function(x):
    return 1 if x >= 0 else 0


def perceptron_and():
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    y = np.array([0, 0, 0, 1])

    weights = np.zeros(2)
    bias = 0
    learning_rate = 0.1

    # Training
    for epoch in range(10):
        for i in range(len(X)):
            weighted_sum = np.dot(X[i], weights) + bias
            prediction = step_function(weighted_sum)

            error = y[i] - prediction

            weights = weights + learning_rate * error * X[i]
            bias = bias + learning_rate * error

    print("AND Gate using Perceptron:")
    for x in X:
        result = step_function(np.dot(x, weights) + bias)
        print(x, "->", result)


# -------------------------------
# Backpropagation Neural Network
# XOR Gate
# -------------------------------

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def backpropagation_xor():

    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    y = np.array([
        [0],
        [1],
        [1],
        [0]
    ])

    np.random.seed(1)

    # Initialize weights
    weights_input_hidden = np.random.uniform(
        -1, 1, (2, 2)
    )

    weights_hidden_output = np.random.uniform(
        -1, 1, (2, 1)
    )

    learning_rate = 0.5

    # Training
    for epoch in range(10000):

        # Forward propagation
        hidden_input = np.dot(X, weights_input_hidden)
        hidden_output = sigmoid(hidden_input)

        output_input = np.dot(
            hidden_output,
            weights_hidden_output
        )

        output = sigmoid(output_input)

        # Calculate error
        error = y - output

        # Backpropagation
        output_delta = (
            error * sigmoid_derivative(output)
        )

        hidden_error = np.dot(
            output_delta,
            weights_hidden_output.T
        )

        hidden_delta = (
            hidden_error *
            sigmoid_derivative(hidden_output)
        )

        # Update weights
        weights_hidden_output += (
            learning_rate *
            np.dot(hidden_output.T, output_delta)
        )

        weights_input_hidden += (
            learning_rate *
            np.dot(X.T, hidden_delta)
        )

    print("\nXOR Gate using Backpropagation:")

    for i in range(len(X)):
        result = output[i][0]

        if result >= 0.5:
            prediction = 1
        else:
            prediction = 0

        print(X[i], "->", prediction)


# -------------------------------
# Main Program
# -------------------------------

perceptron_and()
backpropagation_xor()
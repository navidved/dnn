import numpy as np

input_vector_1 = np.array([1.66, 1.56])
input_vector_2 = np.array([2, 1.5])

weights_1 =  np.array([1.45, -0.66])
bias =  np.array([0.0])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def make_prediction(input_vector, weights, bias):
    layer_1 = np.dot(input_vector, weights) + bias
    layer_2 = sigmoid(layer_1)
    if layer_2 >= 0.5:
        return 1, layer_2
    else:
        return 0, layer_2

active, prediction = make_prediction(input_vector_1, weights_1, bias)
print(f"the predication 1 result is: {active, prediction}")

target = 1
mse = np.square(prediction - target)
print(mse)

active, prediction = make_prediction(input_vector_2, weights_1, bias)
print(f"the predication 2 result is: {active, prediction}")

target = 0
mse = np.square(prediction - target)
print(mse)
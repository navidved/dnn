import numpy as np
import matplotlib.pyplot as plt
from nn_class import NeuralNetwork


# input_vector = np.array([1.66, 1.56])

# learning_rate = 0.1

# neural_network = NeuralNetwork(learning_rate)

# prediction = neural_network.predict(input_vector)

# print(f"weights: {neural_network.weights}, bias: {neural_network.bias}")
# print(f"prediction: {prediction}")


# -------------------------------------------------

input_vectors = np.array(
   [
       [3, 1.5],
       [2, 1],
       [4, 1.5],
       [3, 4],
       [3.5, 0.5],
       [2, 0.5],
       [5.5, 1],
       [1, 1],
   ]
)

targets = np.array([0, 1, 0, 1, 0, 1, 1, 0])

learning_rate = 0.1

neural_network = NeuralNetwork(learning_rate)

print(f"init => weights: {neural_network.weights}, bias: {neural_network.bias}")

training_error = neural_network.train(input_vectors, targets, 10000)

print(f"after train => weights: {neural_network.weights}, bias: {neural_network.bias}")

plt.plot(training_error)
plt.xlabel("Iterations")
plt.ylabel("Error for all training instances")
plt.legend()
plt.savefig("cumulative_error.png")
plt.show()
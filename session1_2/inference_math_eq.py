import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# Build the neural network model
model = Sequential([
    Dense(10, input_shape=(1,), activation='relu'),
    Dense(10, activation='relu'),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')


model.load_weights('math_eq_model')

model.summary()

# # Test the model
# x_test = np.linspace(-5,5, 10).reshape(-1,1)
# y_pred = model.predict(x_test)
# tv_test = 3*x_test**2 + 2*x_test + 1


# # Generate some random data
# np.random.seed(0)
# x_train = np.random.rand(1000,1) * 10 - 5
# y_train = 3*x_train**2 + 2*x_train + 1


# # Plot the results
# plt.scatter(x_train, y_train, color='blue', label='Training Data')
# plt.plot(x_test, y_pred, color='red', label="Predictions")
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Neural Network Prediction')
# plt.legend()
# plt.show()

print(model.predict([3]))
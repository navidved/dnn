import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# Math eq: y= 3*x^2 +2x+1


# Generate some random data
np.random.seed(0)
x_train = np.random.rand(1000,1) * 10 - 5
y_train = 3*x_train**2 + 2*x_train + 1


# Build the neural network model
model = Sequential([
    Dense(10, input_shape=(1,), activation='relu'),
    Dense(10, activation='relu'),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')


# Train the model
model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1)

# Save model
model.save_weights('math_eq_model')

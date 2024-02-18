import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Random data generation
x = np.linspace(0,50,51)
y = x + np.random.random(len(x)) * 10

# plt.scatter(x, y, label="Training Data")
# plt.legend()
# plt.show()

# modelling 
# y_pred = x*w + b
# MSE = (y_pred - y)^2
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1, input_shape=[1]))
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))

model.summary()

history = model.fit(x, y, epochs=100)

# plt.plot(history.history['loss'])
# plt.show()

y_pred = model.predict(x)


layer = model.get_layer(index=0)
print(layer)

weights = layer.get_weights()
print(weights)

w, b = weights[0][0], weights[1]
print(f"w:{w}, b:{b}")

y_pred_ex = w * x + b



# plt.scatter(x, y, label="training data")
# plt.plot(x, y_pred, label="predicted with model", color='c')
# plt.legend()
# plt.show()
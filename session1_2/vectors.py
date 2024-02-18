import matplotlib.pyplot as plt
import numpy as np


# create a single vector
input_vector = np.array([1.72, 1.23])
weights_1 = np.array([1.26, 0])
weights_2 = np.array([2.17, 0.32])


mult_w1_input_manual = input_vector[0] * weights_1[0] + input_vector[1] * weights_1[1] 
print(mult_w1_input_manual)

mult_w1_input_np = np.dot(input_vector, weights_1)
print(mult_w1_input_np)

mult_w2_input_np = np.dot(input_vector, weights_2)
print(mult_w2_input_np)









fig, ax = plt.subplots()

ax.quiver(0,0, input_vector[0], input_vector[1], angles='xy', scale_units='xy', scale=1, color='r')
ax.quiver(0,0, weights_1[0], weights_1[1], angles='xy', scale_units='xy', scale=1, color='b')
ax.quiver(0,0, weights_2[0], weights_2[1], angles='xy', scale_units='xy', scale=1, color='c')


ax.set_xlim([0, 2.5])
ax.set_ylim([0, 2.5])

plt.grid()
plt.show()



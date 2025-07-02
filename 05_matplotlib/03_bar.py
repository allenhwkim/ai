import matplotlib.pyplot as plt
import numpy as np

# 1. bar
x = np.array([1, 2, 3, 4, 5])  # x-axis values
y = np.array([2, 3, 5, 7, 11])  # y-axis values
plt.bar(x, y)  # Create
plt.show()
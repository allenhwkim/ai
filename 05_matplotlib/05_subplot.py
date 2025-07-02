# Subplots allow you to display multiple plots in a single figure.

import matplotlib.pyplot as plt
import numpy as np

# Create a figure and a set of subplots (1 row, 2 columns
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Generate some data
x = np.linspace(0, 10, 100)

# Plot the first subplot
axs[0].plot(x, np.sin(x), label="sin(x)")  # Plot sine wave
axs[0].legend()

# Plot the second subplot
axs[1].plot(x, np.cos(x), label="cos(x)", color="red")  # 'r' is for red color
axs[1].legend()

# Plot the second subplot
axs[2].plot(x, np.sin(x), label="sin(x)")  # Plot sine wave
axs[2].plot(x, np.cos(x), label="cos(x)", color="red")  # 'r' is for red color
axs[2].legend()

plt.tight_layout() # Adjust layout to prevent overlap
plt.show()

# Subplots allow you to display multiple plots in a single figure.

import matplotlib.pyplot as plt
import numpy as np

# Create a figure and a set of subplots (1 row, 2 columns
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Generate some data
x = np.linspace(0, 10, 100)

# Plot the first subplot
axs[0].plot(x, np.sin(x))
axs[0].set_title('Sine Wave')
axs[0].set_xlabel('x')
axs[0].set_ylabel('sin(x)') 

# Plot the second subplot
axs[1].plot(x, np.cos(x), 'r')  # 'r' is for red color
axs[1].set_title('Cosine Wave')
axs[1].set_xlabel('x') 
axs[1].set_ylabel('cos(x)')

plt.tight_layout() # Adjust layout to prevent overlap
plt.show()
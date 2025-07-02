import matplotlib.pyplot as plt
import numpy as np

# 1. line
x = np.linspace(0, 10, 100)  # Generate 100 points from 0 to 10
y = 2 * x + 10  # Compute the sine of each point

plt.plot(x, y)  # Plot the sine wave
plt.ylim(0) # set y-axis range
plt.show()


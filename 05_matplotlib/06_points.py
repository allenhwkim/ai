import matplotlib.pyplot as plt
import numpy as np

ypoints = np.array([3, 8, 1, 10, 5, 7])

plt.plot(ypoints)
plt.plot(ypoints, 'o')  # Plot points as circles
plt.show()
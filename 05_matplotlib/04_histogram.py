import matplotlib.pyplot as plt
import numpy as np

# histogram example
# Histograms are used to show the distribution of a dataset.
data = np.random.randn(1000)  # Generate 1000 random numbers from a normal distribution
plt.hist(data, bins=30, alpha=0.7)  # Create a histogram with 30 bins
plt.show()  # Display the histogram
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

img = Image.open('data/image.png')
img = np.asarray(img)
print(type(img), img.shape)  # <class 'numpy.ndarray'> (480, 640, 3)
imgplot = plt.imshow(img)
# imgplot.set_cmap('hot')
plt.show()

# Resize image to 64x64 pixels
img = Image.open('data/image.png') 
img.thumbnail((64, 64))  # resizes image in-place
img = np.asarray(img)
print(type(img), img.size) # <class 'PIL.Image.Image'> (64, 64)
imgplot = plt.imshow(img)
plt.colorbar(imgplot)
# imgplot.set_alpha(0.5)  # Set transparency
plt.show()
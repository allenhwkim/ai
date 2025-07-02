"""
A tensor is a multi-dimensional array, similar to NumPy arrays but optimized for GPU acceleration.
"tensor" comes from the Latin word "tendere," meaning "to stretch." 
"intense", "tension", "extend", and "tendency" are all derived from the same root.

Tensors comes from physics, where it is used to represent physical quantities that have both magnitude and direction.

https://phys.libretexts.org/@api/deki/files/3862/clipboard_efc45921bb2c4eaff288d52f56acbe7c8?revision=1&size=bestfit&width=575&height=335
Scalar: A single number, represented as a 0D tensor. 50 m
Vector: A one-dimensional array of numbers, represented as a 1D tensor. 
  e.g. [5.0, 0.0] m/s (5 m/s in the x-direction, 0 m/s in the y-direction)
Matrix: A two-dimensional array of numbers, represented as a 2D tensor.
  e.g. [ [2.1, 0.0], [0.0, 0.0] ] m/s^2 (acceleration in the x, y directions)
    x-acceleration(to the east) caused by x, y [2.1, 0.0], 
    y-acceleration(to the north/south) caused by x, y motion [0, 0],

To work with tensors, PyTorch is used, a popular open-source machine learning library.

# PyTorch Tensors: Introduction and Basic Operations
This code provides an introduction to PyTorch tensors, covering their creation, basic operations, properties,
and conversion between NumPy arrays and PyTorch tensors. It also includes examples of reshaping tensors
and performing basic autograd operations for gradient computation.
"""

import torch
import numpy as np

# 1. Introduction to Tensors, Scalar, Vector, Matrix, and 3D Tensors
scalar = torch.tensor(5)  # A single number (0D tensor)
print("Scalar:", scalar) 
print(scalar.dim(), scalar.shape, scalar.dtype) # 0 torch.Size([]) torch.int64
vector = torch.tensor([1, 2, 3])  # A one-dimensional tensor (1D tensor)
print("Vector:", vector)
print(vector.dim(), vector.shape, vector.dtype)  # 1 torch.Size([3]) torch.int64 
matrix = torch.tensor([[1, 2], [3, 4]])  # A two-dimensional tensor (2D tensor)
print("Matrix:", matrix)
print(matrix.dim(), matrix.shape, matrix.dtype)  # 2 torch.Size([2, 2]) torch.int64
three_d_tensor = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # A three-dimensional tensor (3D tensor)
print("3D Tensor:", three_d_tensor)
print(three_d_tensor.dim(), three_d_tensor.shape, three_d_tensor.dtype)  # 3 torch.Size([2, 2, 2]) torch.int64

# 2. Basic Tensor Operations (Addition, Multiplication, Matrix Multiplication)
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
print("Addition:", a + b)  # [5, 7, 9]
print("Subtraction:", a - b)  # [-3, -3, -3]
print("Multiplication:", a * b) # [4, 10, 18]
print("Division:", a / b) # [0.25, 0.4, 0.5]
print("Matrix Multiplication:", torch.matmul(a.unsqueeze(0), b.unsqueeze(1))) # [[32]] # 1x3 matrix multiplied by 3x1 matrix results in a 1x1 matrix
print("Element-wise Exponentiation:", a ** 2)  # [1, 4, 9]
print("Element-wise Square Root:", torch.sqrt(a.float()))  # [1.0, 1.4142, 1.7321]
print("Element-wise Logarithm:", torch.log(b.float()))  # [1.3863, 1.6094, 1.7918]
print("Element-wise Sine:", torch.sin(b.float()))  # [-0.7568, -0.9589, -0.2794]
print("Element-wise Cosine:", torch.cos(b.float()))  # [-0.6536, -0.2837, 0.9602]
print("Element-wise Tangent:", torch.tan(b.float()))  # [1.1578, 3.3805, -0.2910]
print("Element-wise Maximum:", torch.maximum(a, b))  # [4, 5, 6]
print("Element-wise Minimum:", torch.minimum(a, b))  # [1, 2, 3]
print("Element-wise Absolute Value:", torch.abs(a - b))  # [3, 3, 3]

# 3. Tensor Properties (Shape, Data Type, Device)
a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)  # 2D tensor with float data type
print("Shape:", a.shape)  # torch.Size([2, 2])
print("Data Type:", a.dtype)  # torch.float32
print("Device:", a.device)  # cpu (or cuda if using GPU)

# 4. Generating tensors (zeros, ones, random numbers)
zeros = torch.zeros((2, 3))  # 2x3 tensor filled with zeros
print("Zeros Tensor:", zeros) # [[0., 0., 0.], [0., 0., 0.]]
ones = torch.ones((2, 3))  # 2x3 tensor filled with ones
print("Ones Tensor:", ones)  # [[1., 1., 1.], [1., 1., 1.]]
random_tensor = torch.rand((2, 3))  # 2x3 tensor with random numbers between 0 and 1
print("Random Tensor:", random_tensor)  # e.g., [[0.1234, 0.5678, 0.9101], [0.2345, 0.6789, 0.0123]]

# 5. Converting between NumPy and PyTorch Tensors
numpy_array = np.array([[1, 2], [3, 4]], dtype=np.float32)  # NumPy array
print("NumPy Array:", numpy_array)  # [[1. 2.] [3. 4.]]
pytorch_tensor = torch.from_numpy(numpy_array)  # Convert NumPy array to PyTorch tensor
print("PyTorch Tensor from NumPy Array:", pytorch_tensor)  # tensor([[1., 2.], [3., 4.]])
numpy_array_from_tensor = pytorch_tensor.numpy()  # Convert PyTorch tensor back to NumPy array
print("NumPy Array from PyTorch Tensor:", numpy_array_from_tensor)  # [[1. 2.] [3. 4.]] 

# 6. Reshaping Tensors
original = torch.tensor([[1, 2, 3, 4, 5, 6]])
print("Original Tensor:", original)  # tensor([[1, 2, 3, 4, 5, 6]])
print("Original shape:", original.shape)  # torch.Size([1, 6])
reshaped = original.view(2, 3)  # Reshape to 2 rows and 3 columns
print("Reshaped Tensor:", reshaped)  # tensor([[1, 2, 3], [4, 5, 6]])
print("Reshaped shape:", reshaped.shape)  # torch.Size([2, 3])

# 7. squeeze and Unsqueeze
original = torch.tensor([1, 2, 3, 4, 5, 6])  # 1D tensor
print("Original Tensor:", original, original.shape) # tensor([1, 2, 3, 4, 5, 6]) torch.Size([6])
unsqueezed = original.unsqueeze(0)  # Add a new dimension at index 0
print("Unsqueezed Tensor:", unsqueezed, unsqueezed.shape)  # tensor([[1, 2, 3, 4, 5, 6]]) torch.Size([1, 6])
unsqueezed = unsqueezed.unsqueeze(0)  # Add a new dimension at index 0
print("Unsqueezed Tensor:", unsqueezed, unsqueezed.shape)  #tensor([[[1, 2, 3, 4, 5, 6]u]]) torch.Size([1, 1, 6])
squeezed = unsqueezed.squeeze(0)  # Add a new dimension at index 1
print("Squeezed Tensor:", squeezed, squeezed.shape) # tensor([[1, 2, 3, 4, 5, 6]]) torch.Size([1, 6])
squeezed = squeezed.squeeze(0)  # Remove the dimension at index 0
print("Squeezed Tensor:", squeezed, squeezed.shape)  # tensor([1, 2, 3, 4, 5, 6]) torch.Size([6])
squeezed = squeezed.squeeze(0)  # Remove the dimension at index 0
"""
Broadcasting is a way to perform an operation 
between similarily-shaped ensors.

e.g. (2, 2) * (1,2) = (2, 2) # shape1 * shape2 = shape1
  [[1,2], [3,4]] * [[5,6]] = [[5,12], [15,24]]
  [1*5, 2*6], [3*5, 4*6]] = [[5, 12], [15, 24]]
"""
import numpy as np
import torch

# 1. Add vector to each row of the matrix
a = torch.tensor([[1,2], [3,4]]) # shape (2, 2)
b = torch.tensor([5,6])        # shape (1, 2)
result = a + b  # Adding by rows of b
print(result)   # tensor([[ 6,  8], [ 8, 10]]) 
# [1,2] + [5,6] = [1+5, 2+6] = [6, 8]
# [3,4] + [5,6] = [3+5, 4+6] = [8, 10]

# 2. Multiply matrix by vector
a = torch.tensor([[1,2], [3,4]]) # shape (2, 2)
b = torch.tensor([5,6]).unsqueeze(0) # shape (1, 2)
result = a * b  # multiply by each element of b
print(result)   # tensor([[ 5, 12], [15, 24]])
# [1,2] * [5,6] = [1*5, 2*6] = [5, 12]
# [3,4] * [5,6] = [3*5, 4*6] = [15, 24]

# 3. Multiply vector with different shape
a = torch.tensor([[1,2,3], [4,5,6]]) # shape (2, 3)
b = torch.tensor([[7], [8]]) # shape (2, 1)
result = a * b  # multiply by each element of b
print(result)   # tensor([[ 7, 14, 21], [32, 40, 48]])
# [1,2,3] * [7,8] = [1*7, 2*8, 3*7] = [7, 14, 21]
# [4,5,6] * [7,8] = [4*7, 5*8, 6*7] = [32, 40, 48]
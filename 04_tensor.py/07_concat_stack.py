"""
Concatenation (torch.cat): 
    Combines multiple tensors along an existing dimension, 
    increasing the size of that dimension.
Stacking (torch.stack): 
    Combines multiple tensors along a new dimension, 
    creating a new axis without merging the data directly.
"""

import torch

# 1. Concatenation
tensor_a = torch.tensor([[1, 2], [3, 4]])
tensor_b = torch.tensor([[5, 6], [7, 8]])
concat_dim0 = torch.cat((tensor_a, tensor_b), dim=0)
print(concat_dim0) # tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
concat_dim1 = torch.cat((tensor_a, tensor_b), dim=1)
print(concat_dim1) # tensor([[1, 2, 5, 6], [3, 4, 7, 8]])

# 2. Stacking
# Note: Stacking creates a new dimension, while concatenation merges along an existing one.
tensor_a = torch.tensor([[1, 2], [3, 4]]) # (2, 2)
tensor_b = torch.tensor([[5, 6], [7, 8]]) # (2, 2)
stack_dim0 = torch.stack((tensor_a, tensor_b), dim=0)
print(stack_dim0) # tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]) # (2, 2, 2)
stack_dim1 = torch.stack((tensor_a, tensor_b), dim=1)
print(stack_dim1) # tensor([[[1, 2], [5, 6]], [[3, 4], [7, 8]]]) # (2, 2, 2)
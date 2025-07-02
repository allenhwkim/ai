"""
View: 
    A tensor that shares the same underlying data as the original tensor 
    but presents it differently (e.g., reshaped). 
    Modifying a view modifies the original tensor.
Copy: 
    A new tensor with independent data. 
    Modifying a copy does not affect the original tensor.
"""

import torch

# 1. View: Reshape the tensor
tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
view_tensor = tensor.view(3, 2)
print(tensor)        # [[1, 2, 3], [4, 5, 6]]
print(view_tensor) # [[1, 2], [3, 4], [5, 6]]
view_tensor[0, 0] = 99
print(tensor)        # [[99, 2, 3], [4, 5, 6]]
print(view_tensor) # [[99, 2], [3, 4], [5, 6]]

# 2. Copy: Create a copy of the tensor
tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
copy_tensor = tensor.clone()
print(copy_tensor)  # [[1, 2, 3], [4, 5, 6]]
copy_tensor[0, 0] = 99
print(tensor)       # [[1, 2, 3], [4, 5, 6]]
print(copy_tensor)  # [[99, 2, 3], [4, 5, 6]]

# 3. Copy: detach a tensor
x = torch.tensor([2.0, 3.0], requires_grad=True)
print(x) # tensor([2., 3.], requires_grad=True) 
print(x.requires_grad) # True
y = x * 2 + 1  
print(y)  # tensor([5., 7.], grad_fn=<AddBackward0>)
print(y.requires_grad)  # True, as y is derived
z = y.sum() # tensor(12., grad_fn=<SumBackward0>)
print(z)  # tensor(12., grad_fn=<SumBackward0>)
z.backward() # Compute gradients with backward()
print("\nGradient of x (dz/dx):\n", x.grad) # tensor([2., 2.]), as dz/dx = 2 for each element
detached_x = x.detach()
print(detached_x) # tensor([2., 3.]) # no requires_grad

# A tensor with requires_grad=False
x = torch.tensor([2.0, 3.0]) # tensor without requires_grad
print(x) # tensor([2., 3.])
print(x.requires_grad) # False, as a does not track gradients
y = x * 2 + 1 
print(y) # tensor([5., 7.])
z = x.sum() 
print(z)  # tensor(5.) # x = [2, 3]
# This would raise an error if uncommented, as no graph is built
# c.backward()
# print(a.grad)  # None, as no gradients are tracked

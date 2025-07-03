"""
The requires_grad attribute in PyTorch determines whether operations on a tensor 
should be tracked for automatic differentiation. 

If requires_grad=True, PyTorch will record all operations on that tensor, 
allowing you to compute gradients with respect to it during backpropagation (using .backward()). 

backpropagation describes the process of computing all gradients 
in a network by moving backward from outputs to inputs.
"""
import torch

# 3. Copy: detach a tensor
x = torch.tensor([2.0, 3.0], requires_grad=True)
print(x) # tensor([2., 3.], requires_grad=True) 
print(x.requires_grad) # True
y = x*2 + 1  
print(y)  # tensor([5., 7.], grad_fn=<AddBackward0>)
print(y.requires_grad)  # True, as y is derived
z = y.sum() # tensor(12., grad_fn=<SumBackward0>)
print(z)  # tensor(12., grad_fn=<SumBackward0>)
z.backward() # Compute gradients with backward()
"""
backward pass from x -> y -> z
0. z.backward() is called
1. Compute gradients of z with respect to y
2. Compute gradients of y with respect to x
3. Store gradients in x.grad
"""
print(x.grad) # Gradient tensor([2., 2.]), as dz/dx = 2 for each element


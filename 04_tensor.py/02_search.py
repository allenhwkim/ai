import torch

# 1. search with index 
x = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(x[1, 2]) # tensor(7)  # second row, third column
# rows 0, 1, columns 1, 2, 3 (not including the last number)
print(x[0:2, 1:4]) # tensor([[2, 3, 4], [6, 7, 8]]) 

# 2. search with condition
x = torch.tensor([1, 2, 3, 4, 5, 6])
print(x[x > 3]) # tensor([4, 5, 6])  # elements greater than 3
print(x[x < 3]) # tensor([1, 2])  # elements less than 3

# 3. search with multiple conditions
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(x[(x > 3) & (x < 8)]) # tensor([4, 5, 6, 7])  # elements greater than 3

# 4. find indices of elements that match a condition
x = torch.tensor([1, 2, 3, 4, 5, 6])
indices = (x > 3).nonzero(as_tuple=True)[0]
print(indices) # tensor([3, 4, 5])  # indices of elements greater than 3



"""
Statistics module for tensor operations.
To compute mean, variance, and standard deviation for tensors. 
"""
import torch

a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)  # shape (2, 3)
b = torch.tensor([7, 8, 9], dtype=torch.float32)

# 1. Mean
mean_a = torch.mean(a, dim=0)  # Mean along the first dimension (columns)
mean_b = torch.mean(b)         # Mean of the vector b
mean_c = torch.mean(a, dim=1)  # Mean along the second dimension (rows)
print("Mean of a:", mean_a)  # tensor([2.5000, 3.5000, 4.5000])
# 2.5 = (1 + 4) / 2, 3.5 = (2 + 5) / 2, 4.5 = (3 + 6) / 2
print("Mean of b:", mean_b)  # tensor(8.0000)
print("Mean of a along rows:", mean_c)  # tensor([2.0000, 5.0000])
# 2.0 = (1 + 2 + 3) / 3, 5.0 = (4 + 5 + 6) / 3

# 2. Variance (the average of the squared differences from the mean)
a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)  # shape (2, 3)
var_col = torch.var(a, dim=0, unbiased=False) 
print("Variance of a along cols:", var_col) # tensor([2.5000, 2.5000, 2.5000])
# 2.5 = ((1-2.5)^2 + (4-2.5)^2) / 2, same for other columns
# 2.5 = ((2-3.5)^2 + (5-3.5)^2) / 2, same for other columns
# 2.5 = ((3-4.5)^2 + (6-4.5)^2) / 2, same for other columns

var_row = torch.var(a, dim=1, unbiased=False)
print("Variance of a along rows:", var_row) # tensor([0.6667, 0.6667])
# 0.6667 = ((1-2)^2 + (2-2)^2 + (3-2)^2) / 3
# 0.6667 = ((4-5)^2 + (5-5)^2 + (6-5)^2) / 3

a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32) 
var_a = torch.var(a, unbiased=False)
print("mean of a:", torch.mean(a))  # tensor(3.5000)
print("Variance of b:", var_a) # tensor(2.9167) 
# 2.9167 = ((1 - 3.5)^2 + (2 - 3.5)^2 + (3 - 3.5)^2 + ...))

# 3. Standard Deviation (the square root of the variance)
a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32) 
std_a = torch.std(a, unbiased=False)
print("Standard Deviation of a:", std_a)  # tensor(1.7078)

# 5. Minimum and Maximum
a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32) 
min_a = torch.min(a)  # Minimum value
max_a = torch.max(a)  # Maximum value
print("Minimum of a:", min_a)  # tensor(1.0000)
print("Maximum of a:", max_a)  # tensor(6.0000)

# 6. Sum
a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32) 
sum_a = torch.sum(a)  # Sum of all elements
print("Sum of a:", sum_a)  # tensor(21.0000)
# 21 = 1 + 2 + 3 + 4 + 5 + 6

# 7. Product
a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32) 
prod_a = torch.prod(a)  # Product of all elements
print("Product of a:", prod_a)  # tensor(720.0000)
# 720 = 1 * 2 * 3 * 4 * 5 * 6

# 8. Median
a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32) 
median_a = torch.median(a)  # Median
print("Median of a:", median_a)  # tensor(3.5000)
# 3.5 = (3 + 4) / 2, since the elements are sorted as [1, 2, 3, 4, 5, 6]

# 9. unique elements
a = torch.tensor([1, 2, 2, 3, 4, 4, 5], dtype=torch.float32) 
unique_a = torch.unique(a)
print("Unique elements in a:", unique_a)  # tensor([1., 2., 3., 4., 5.])
# Unique elements are [1, 2, 3, 4, 5] 

# 10. Mode (most frequent value)
a = torch.tensor([1, 2, 2, 3, 4, 4, 5], dtype=torch.float32) 
mode_a = torch.mode(a)
print("Mode of a:", mode_a.values)  # tensor(2.)
# Mode is 2, since it appears most frequently (twice)
print("Count of mode:", mode_a.count)  # tensor(2)
# Count of mode is 2, since 2 appears twice







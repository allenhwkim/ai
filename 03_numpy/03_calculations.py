"""
With numpy, you can perform a wide range of calculations on arrays, including element-wise operations, statistical calculations, and linear algebra operations.
"""

import numpy as np

# 1. Element-wise addition
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
result_add = arr1 + arr2
print("Element-wise addition:", result_add)  # Output: [5 7 9

# 2. Element-wise subtraction
result_sub = arr1 - arr2
print("Element-wise subtraction:", result_sub)  # Output: [-3 -3 -3]

# 3. Element-wise multiplication
result_mul = arr1 * arr2
print("Element-wise multiplication:", result_mul)  # Output: [ 4 10 18]

# 4. Element-wise division
result_div = arr1 / arr2
print("Element-wise division:", result_div)  # Output: [0.25 0.4 0.5]

# 5. Element-wise exponentiation
result_exp = arr1 ** 2
print("Element-wise exponentiation:", result_exp)  # Output: [1 4 9]
result_exp = arr1 ** arr2
print("Element-wise exponentiation with arr2:", result_exp)  # Output: [ 1 32 729]

# 6. Element-wise square root
result_sqrt = np.sqrt(arr1)
print("Element-wise square root:", result_sqrt)  # Output: [1. 1.41421356 1.73205081]

# 7. Element-wise logarithm
result_log = np.log(arr2)
print("Element-wise logarithm:", result_log)  # Output: [1.38629436 1.60943791 1.79175947]

# 8. Element-wise sine
result_sin = np.sin(arr2)
print("Element-wise sine:", result_sin)  # Output: [ -0.7568025  -0.95892427  -0.2794155 ]

# 9. Element-wise cosine
result_cos = np.cos(arr2)
print("Element-wise cosine:", result_cos)  # Output: [-0.65364362 -0.28366219  0.96017029]

# 10. Element-wise tangent
result_tan = np.tan(arr2)
print("Element-wise tangent:", result_tan)  # Output: [ 1.15782128  3.38051501 -0.29100619]

# 11. Element-wise maximum
result_max = np.maximum(arr1, arr2)
print("Element-wise maximum:", result_max)  # Output: [4 5 6]

# 12. Element-wise minimum
result_min = np.minimum(arr1, arr2)
print("Element-wise minimum:", result_min)  # Output: [1 2 3]

# 13. Element-wise absolute value
result_abs = np.abs(arr1 - arr2)
print("Element-wise absolute value:", result_abs)  # Output: [3 3 3]

# 14. Element-wise comparison
result_gt = arr1 > arr2
print("Element-wise greater than:", result_gt)  # Output: [False False False]

# 15. Element-wise logical operations
result_and = np.logical_and(arr1 > 1, arr2 < 6)
print("Element-wise logical AND:", result_and)  # Output: [False False  True]

# 16. Element-wise logical OR
result_or = np.logical_or(arr1 < 2, arr2 > 5)
print("Element-wise logical OR:", result_or)  # Output: [ True  True False]

# 17. Element-wise logical NOT
result_not = np.logical_not(arr1 > 2)
print("Element-wise logical NOT:", result_not)  # Output: [ True  True False]




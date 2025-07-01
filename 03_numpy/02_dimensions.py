"""
Dimensions in NumPy refer to the number of axes or directions in which data can be organized. A one-dimensional array is like a list, a two-dimensional array is like a table, and a three-dimensional array can be thought of as a cube of data.
"""

import numpy as np

# 1. One-dimensional array
arr1d = np.array([1, 2, 3])
print(arr1d)
print(arr1d.shape)  # (3,)
print(arr1d[0])  # first element 1
print(arr1d[-1])  # last element 3
print(arr1d[1:2]) # index 1 to index 3, [2 3]
for x in arr1d:
    print(x)  # 1, 2, 3

# 2. Two-dimensional array
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(arr2d)
print(arr2d.shape)  # (3, 3)
print(arr2d[1, 2])  # index 1, index 2, output: 6 
print(arr2d[0])  # first element [1 2 3]
print(arr2d[-1])  # last element [7 8 9]
print(arr2d[:, -1]) # last element of each row, output: [3 6 9] 
for row in arr2d:
    for col in row:
        print(col) # 1, 2, 3, 4, 5, 6, 7, 8, 9

# 3. Three-dimensional array
arr3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
"""
        +--------+
        |  9 |10 |
        +----+---+
        | 11 |12 |
    +-------+
    | 5 | 6 |
    +---+---+
    | 7 | 8 |
+-------+
| 1 | 2 | 
+---+---+ 
| 3 | 4 | 
+-------+ 
"""
print(arr3d) 
print(arr3d.shape)  # (3, 2, 2)
print(arr3d[1, 0, 1])  # 2nd slice, 1st row, 2nd column, output: 6
print(arr3d[0])  # first slice [[1 2] [3 4]]
print(arr3d[-1])  # last slice [[ 9 10] [11 12]]
print(arr3d[:, -1, :]) # every slice, last row, all columns, output: [[ 3  4] [ 7  8] [11 12]]
for slice in arr3d:
    for row in slice:
        for col in row:
            print(col)  # 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 

# 4. 1D to 2D conversion
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
arr_2d = arr.reshape(3, 4)  # Reshape to 3 rows and 4 columns
print(arr_2d)
print(arr.shape, arr_2d.shape)  # (12,) (3, 4)

# 5. 1D to 3D conversion
arr_3d = arr.reshape(2, 3, 2)  # Reshape to 2 slices, 3 rows, and 2 columns
print(arr_3d)
print(arr.shape, arr_3d.shape)  # (12,) (2, 3, 2)

# 6. 3D to 1D conversion
arr_1d_from_3d = arr_3d.flatten()  # Flatten the 3D array to 1D
print(arr_1d_from_3d)
print(arr_3d.shape, arr_1d_from_3d.shape)  # (2, 3, 2) (12,)
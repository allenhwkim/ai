"""
NumPy: Numerical Python
====================
Arrays are very frequently used in data science, where speed is very important.
NumPy is a Python library used for working with arrays.

Why not list for arrays? 
-----------------------
List are slow to process. Numpy is 50x faster than lists. 

ndarray? 
-----------------------
The array object in NumPy is called ndarray, which stands for n-dimensional array.
  - a fast and flexible container for large data sets in Python.
  - a table of elements (usually numbers) all of the same type
  - indexed by a tuple of non-negative integers.
"""

# 1. Get started with NumPy
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
print("NumPy Array:", arr, type(arr)) # NumPy Array: [1 2 3 4 5] <class 'numpy.ndarray'>
print("Array Type:", arr.dtype)  # Array Type: int64

# 2. Create a NumPy array
#   - from a list
arr_from_list = np.array([1, 2, 3, 4, 5], dtype='i4')
print("Array from list:", arr_from_list)
print("Array Type from list:", arr_from_list.dtype) # Array Type from list: int32 
#   - from a tuple
arr_from_tuple = np.array((1, 2, 3, 4, 5), dtype=float)
print("Array from tuple:", arr_from_tuple)
print("Array Type from tuple:", arr_from_tuple.dtype)  # Array Type from tuple: float64
#   - from a range
arr_from_range = np.array(range(1, 6), dtype=int)
print("Array from range:", arr_from_range)
print("Array Type from range:", arr_from_range.dtype)  # Array Type from range: int64
#   - from a string
arr_from_string = np.array(list("Hello"), dtype='U1')
print("Array from string:", arr_from_string)  # Array from string: ['H' 'e' 'l' 'l' 'o']
print("Array Type from string:", arr_from_string.dtype)  # Array Type from string: <U1
#   - from a boolean
arr_from_bool = np.array([True, False, True], dtype=bool)
print("Array from boolean:", arr_from_bool)  # Array from boolean: [ True False  True]
print("Array Type from boolean:", arr_from_bool.dtype)  # Array Type from boolean: bool
#   - from a complex number
arr_from_complex = np.array([1+2j, 3+4j, 5+6j], dtype=complex)
print("Array from complex number:", arr_from_complex)  # Array from complex number: [1.+2.j 3.+4.j 5.+6.j]
print("Array Type from complex number:", arr_from_complex.dtype)  # Array Type from complex number: complex128


"""
Tuples are immutable sequences with no fixed order.
Tuples are faster than lists and can be used as dictionary keys.
Tuples are defined using parentheses `()`. 
"""

# 1. Define a tuple
my_tuple = (1, 2, 3)
print("my_tuple:", my_tuple)  # (1, 2, 3)

# 2. Access elements
print("my_tuple[0]:", my_tuple[0])  # 1
print("my_tuple[1]:", my_tuple[1])  # 2
print("my_tuple[-1]:", my_tuple[-1])  # 3 (last element)
print("my_tuple[-2]:", my_tuple[-2])  # 2 (second last element)

# 3. Slicing
print("my_tuple[1:3]:", my_tuple[1:3])  # (2, 3) (elements at index 1 and 2)
print("my_tuple[:3]:", my_tuple[:3])  # (1, 2, 3)
print("my_tuple[2:]:", my_tuple[2:])  # (3)
print("my_tuple[-2:]:", my_tuple[-2:])  # (2, 3) (last two elements)
print("my_tuple[::2]:", my_tuple[::2])  # (1, 3) (every second element)
print("my_tuple[::-1]:", my_tuple[::-1])  # (3, 2, 1) (reversed tuple)

# 4. Tuple methods
print("my_tuple.count(2):", my_tuple.count(2))  # 1
print("my_tuple.index(2):", my_tuple.index(2))  # 1 # (index of first occurrence of 2)
# print("my_tuple.append(4):", my_tuple.append(4))  # Error: 'tuple' object has no attribute 'append' 
# print("my_tuple.remove(2):", my_tuple.remove(2))  # Error: 'tuple' object has no attribute 'remove' 
# print("my_tuple.pop():", my_tuple.pop())  # Error: 'tuple' object has no attribute 'pop'

# 5. Tuple unpacking
my_tuple = (1, 2, 3)
a, b, c = my_tuple
print("a, b, c: ", a, b, c)  # 1 2 3 

# 6. Tuple concatenation
tuple1 = (1, 2)
tuple2 = (3, 4)
concatenated = tuple1 + tuple2
print("concatenated:", concatenated)  # (1, 2, 3, 4)

# 7. Tuple repetition
repeated = tuple1 * 3
print("repeated:", repeated)  # (1, 2, 1, 2, 1, 2)

# 8. Tuple as dictionary keys
my_dict = {(1, 2): "value1", (3, 4): "value2"}
print("my_dict:", my_dict)  # {(1, 2): 'value1', (3, 4): 'value2'}  

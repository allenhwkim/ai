# 1. Define a list
a = [1, 2, 3, 4, 5]
b = list((1, 2, 3, 4, 5))  # Using list constructor
c = list(range(1, 6))  # Using range to create a list
print('a, b, c: ', a, b, c) # [1, 2, 3, 4, 5] [1, 2, 3, 4, 5] [1, 2, 3, 4, 5]

# 2. Access elements
print('a[0]:', a[0])  # 1
print('a[1]:', a[1])  # 2
print('a[-1]:', a[-1])  # 5 (last element)
print('a[-2]:', a[-2])  # 4 (second last element)

# 3. Slicing
print('a[1:3]:', a[1:3])  # [2, 3] (elements at index 1 and 2)
print('a[:3]:', a[:3])  # [1, 2, 3] (first three elements)
print('a[2:]:', a[2:])  # [3, 4, 5] (elements from index 2 to the end)
print('a[-2:]:', a[-2:])  # [4, 5] (last two elements)
print('a[::2]:', a[::2])  # [1, 3, 5] (every second element)
print('a[::-1]:', a[::-1])  # [5, 4, 3, 2, 1] (reversed list)

# 4. List methods
a.append(6)
print('a:', a)  # [1, 2, 3, 4, 5, 6]
a.extend([7, 8])
print('a after extend:', a)  # [1, 2, 3, 4, 5, 6, 7, 8]
a.insert(0, 0)
print('a after insert:', a)  # [0, 1, 2, 3, 4, 5, 6, 7, 8]
a.remove(0)
print('a after remove:', a)  # [1, 2, 3, 4, 5, 6, 7, 8]
a.pop()  # Remove and return the last element
print('a after pop:', a)  # [1, 2, 3, 4, 5, 6, 7]
print('a.index(3):', a.index(3))  # 2 (index of first occurrence of 3)
print('a.count(3):', a.count(3))  # 1 (count of occurrences of 3)
a.sort()  # Sort the list in ascending order
print('a after sort:', a)  # [1, 2, 3, 4, 5, 6, 7]
a.reverse()  # Reverse the list
print('a after reverse:', a)  # [7, 6, 5, 4, 3, 2, 1]
print('a copy:', a.copy())  # Create a shallow copy of the list
print('a clear:', a.clear())  # Clear the list
print('a after clear:', a)  # []  

# 5. List comprehension
a = [1, 2, 3, 4, 5]
squared = [x ** 2 for x in range(1, 6)]
print('squared:', squared)  # [1, 4, 9, 16, 25]
even_numbers = [x for x in a if x % 2 == 0]
print('even_numbers:', even_numbers)  # [2, 4]

# 6. Flattening a list of lists
matrix = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
flattened = sum(matrix, [])
print('flattened:', flattened)  # [1, 2, 3, 4, 5, 6, 7, 8, 9]

# 7. Unpacking a list
my_list = [1, 2, 3]
a, b, c = my_list
print('a, b, c:', a, b, c)  # 1 2 3

# 8. List cannot perform union, intersection, or difference like sets
# However, you can use set operations to achieve similar results
list1 = [1, 2, 3]
list2 = [3, 4, 5]
# print("list1 | list2:", list1 | list2) # TypeError: unsupported operand type(s) for |: 'list' and 'list'
# print("list1 & list2:", list1 & list2) # TypeError: unsupported operand type(s) for &: 'list' and 'list'
# print("list1 - list2:", list1 - list2) # TypeError: unsupported operand type(s) for -: 'list' and 'list'
print("set(list1) | set(list2):", set(list1) | set(list2))  # {1, 2, 3, 4, 5} (union)
print("set(list1) & set(list2):", set(list1) & set(list2))  # {3} (intersection)
print("set(list1) - set(list2):", set(list1) - set(list2))  # {1, 2} (
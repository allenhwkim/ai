"""
Set data type in Python is an unordered collection of unique elements. Sets are mutable, meaning you can add or remove elements after creation. They are useful for membership testing and eliminating duplicate entries.
Sets are defined using curly braces `{}` or the `set()` constructor.
"""

# 1. Define a set
my_set = {1, 2, 3, 4, 5}
print("my_set:", my_set)  # {1, 2, 3, 4, 5}

# 2. Access elements (Note: Sets do not support indexing)
print("Is 3 in my_set?", 3 in my_set)  # True

# 3. Set methods
print("my_set after add 6:", my_set.add(6))  # {1, 2, 3, 4, 5, 6}
print("my_set after remove 6:", my_set.remove(6))  # {1, 2, 3, 4, 5}
print("my_set after discard 6:", my_set.discard(6))  # {1, 2, 3, 4, 5}
popped_element = my_set.pop()
print("Popped element:", popped_element)  # Could be any element from the set
print("my_set after pop:", my_set)  # Remaining elements after pop 
updated_set = my_set.copy()  # Create a shallow copy of the set
print("Updated set:", updated_set)  # {1, 2, 3, 4, 5}

# 4. Set operations
set_a = {1, 2, 3}
set_b = {3, 4, 5}
union_set = set_a | set_b
print("Union of set_a and set_b:", union_set)  # {1, 2, 3, 4, 5}
intersection_set = set_a & set_b
print("Intersection of set_a and set_b:", intersection_set)  # {3}
difference_set = set_a - set_b
print("Difference of set_a and set_b:", difference_set)  # {1, 2}
symmetric_difference_set = set_a ^ set_b
print("Symmetric Difference of set_a and set_b:", symmetric_difference_set)  # {1, 2, 4, 5}

# 5. Set comprehension
squared_set = {x ** 2 for x in range(1, 6)}
print("Squared set:", squared_set)  # {1, 4, 9, 16, 25} 

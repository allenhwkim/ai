"""
Dictionary in Python is a mutable, unordered collection of key-value pairs. 
Each key must be unique and immutable, while values can be of any type.
Dictionaries are defined using curly braces `{}` or the `dict()` constructor.
"""

# 1. Define a dictionary
my_dict = {"key1": "value1", "key2": "value2"}
print("my_dict:", my_dict)  # {'key1': 'value1', 'key2': 'value2'}
# my_dict = dict(key1="value1", key2="value2")  # Using dict constructor

# 2. Access elements
print("my_dict['key1']:", my_dict["key1"])  # value1
print("my_dict['key2']:", my_dict["key2"])  # value2
# print("my_dict['key3']:", my_dict["key3"])  # KeyError: 'key3' not found

# 3. Dictionary methods
my_dict["key3"] = "value3"  # Add a new key-value pair
print("my_dict after adding key3:", my_dict)  # {'key1': 'value1', 'key2': 'value2', 'key3': 'value3'}
my_dict["key1"] = "new_value1"  # Update an existing key
print("my_dict after updating key1:", my_dict)  # {'key1': 'new_value1', 'key2': 'value2', 'key3': 'value3'}
del my_dict["key2"]  # Remove a key-value pair
print("my_dict after deleting key2:", my_dict)  # {'key1': 'new_value1', 'key3': 'value3'}
popped_value = my_dict.pop("key3")  # Remove and return the value of a key
print("Popped value:", popped_value)  # value3
print("my_dict after popping key3:", my_dict)  # {'key1': 'new_value1'}
print("Keys in my_dict:", my_dict.keys(), type(my_dict.keys())) # dict_keys(['key1']) <class 'dict_keys'>
print("Values in my_dict:", my_dict.values(), type(my_dict.values()))  # dict_values(['new_value1']) <class 'dict_values'>
print("Items in my_dict:", my_dict.items(), type(my_dict.items()))  # dict_items([('key1', 'new_value1')]) <class 'dict_items>

# 4. Dictionary comprehension
squared_dict = {x: x ** 2 for x in range(1, 6)}
print("squared_dict:", squared_dict)  # {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}
even_dict = {x: x ** 2 for x in range(1, 6) if x % 2 == 0}
print("even_dict:", even_dict)  # {2: 4, 4: 16}

# 5. Merging dictionaries
dict1 = {"a": 1, "b": 2}
dict2 = {"b": 3, "c": 4}
merged_dict = {**dict1, **dict2}  # Merging using unpacking
print("merged_dict:", merged_dict)  # {'a': 1, 'b': 3, 'c': 4}
# Alternatively, you can use the `update()` method
dict1.update(dict2)  # Merging using update method
print("dict1 after update:", dict1)  # {'a': 1, 'b': 3, 'c': 4} 

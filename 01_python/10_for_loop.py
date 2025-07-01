"""
For Loop in python
It is used to iterate over a sequence (like a list, tuple, dictionary, set, or string) or other iterable objects.
"""

# 1: Basic for loop
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(fruit)
# Output: apple banana cherry

# 2: Using range() function
for i in range(5):
    print(i)
# Output: 0 1 2 3 4

# 3: Using range() with start and end
for i in range(2, 6):
    print(i)
# Output: 2 3 4 5

# 4: Using range() with start, end, and step
for i in range(1, 10, 2):
    print(i)
# Output: 1 3 5 7 9

# 5: Loop through a dictionary
my_dict = {"a": 1, "b": 2, "c": 3}
for key, value in my_dict.items():
    print(f"{key}: {value}")
# Output: a: 1 b: 2 c: 3

#6. Loop through a set
my_set = {1, 2, 3}
for item in my_set:
    print(item)
# Output: 1 2 3 (order may vary)

#7. Loop through a string
my_string = "hello"
for char in my_string:
    print(char)
# Output: h e l l o

# 8. Loop through a tuple
my_tuple = (1, 2, 3)
for item in my_tuple:
    print(item)
# Output: 1 2 3

# 9. Nested for loop 
for i in range(3):
    for j in range(2):
        print(f"i: {i}, j: {j}")
# Output:
# i: 0, j: 0
# i: 0, j: 1
# ...
# i: 2, j: 1

# 10. Using else with for loop
for i in range(3):
    print(i)
else:
    print("Loop completed")
# Output: 0 1 2
# Loop completed

# 11. Using break to exit loop
for i in range(5):
    if i == 3:
        break
    print(i)
# Output: 0 1 2 

# 12. Using continue to skip an iteration
for i in range(5):
    if i == 2:
        continue
    print(i)
# Output: 0 1 3 4 

# 13. Using pass to do nothing
for i in range(5):
    if i == 2:
        pass  # Do nothing
    print(i)
# Output: 0 1 2 3 4

# 14. Using for loop with list comprehension
squared = [x**2 for x in range(5)]
print(squared) # [0, 1, 4, 9, 16]


# 15. Using for loop with enumerate
fruits = ["apple", "banana", "cherry"]
for index, value in enumerate(fruits):
    print(f"Index: {index}, Value: {value}")
# Output:
# Index: 0, Value: apple
# Index: 1, Value: banana
# Index: 2, Value: cherry


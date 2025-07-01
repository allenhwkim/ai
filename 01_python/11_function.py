"""
Functions in Python are defined using the `def` keyword, 
followed by the function name and parentheses containing any parameters. 
The body of the function is indented.
"""

# 1. Define a simple function
def greet(name):
    """Function to greet a person by name."""
    print(f"Hello, {name}!")
# Call the function
greet("Allen")  # Hello, Alice!

# 2. Function with return value
def add(a, b):
    """Function to add two numbers and return the result."""
    return a + b
# Call the function and print the result
result = add(5, 3)
print(f"Sum: {result}")  # Sum: 8

# 3. Function with default parameters
def multiply(a, b=2):
    """Function to multiply two numbers with a default value for b."""
    return a * b
print(multiply(4))  # 8 (uses default value for b)
print(multiply(4, 3))  # 12 (uses provided value for b)

# 4. Function with variable number of arguments
def concatenate(*args):
    """Function to concatenate multiple strings."""
    return " ".join(args)
print(concatenate("Hello", "world!"))  # Hello world!

# 5. Function with keyword arguments
def person_info(name, age, **args):
    """Function to display person's information with additional keyword arguments."""
    info = f"Name: {name}, Age: {age}"
    for key, value in args.items():
        info += f", {key}: {value}"
    return info
print(person_info("Alice", 30, city="New York", occupation="Engineer"))
# Name: Alice, Age: 30, city: New York, occupation: Engineer
# print(person_info("Alice", 30, city="New York", "Engineer")) # Error: positional argument follows keyword argument

# 6. Lambda function
square = lambda x: x ** 2
print(square(5))  # 25 (square of 5)

# 7. Tail recursion (not optimized in Python)
def factorial(n, acc=1):
    """Tail recursive function to calculate factorial."""
    if n == 0:
        return acc
    else:
        return factorial(n - 1, n * acc)
print(factorial(5))  # 120 (factorial of 5)
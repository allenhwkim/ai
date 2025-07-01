"""
Exception handling in Python is a way to manage errors and exceptions that occur during the execution of a program. It allows developers to write code that can gracefully handle unexpected situations, rather than crashing the program.
It is done using the `try`, `except`, `else`, and `finally` blocks. 
"""

# 1. Basic try-except block
try:
    result = 10 / 0  # This will raise a ZeroDivisionError
except ZeroDivisionError as e:
    print("Error:", e)

# 2. Catching multiple exceptions
try:
    value = int("abc")  # This will raise a ValueError
except (ValueError, TypeError) as e:
    print("Caught an exception:", e)

# 3. Using finally block
try:
    file = open("example.txt", "r")
    content = file.read()   # This will raise an error if the file does not exist 
except FileNotFoundError:
    print("File not found!")
finally:
    print("This block always executes, whether an exception occurred or not.")
    # If the file was opened, it should be closed here
    try:
        file.close()
    except NameError:
        pass
    
# 4. Raising exceptions
def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero!")
    return a / b 
try:  
    result = divide(10, 0)
except ValueError as e:
    print("Caught an exception:", e)

# 5. Custom exception classes
class CustomError(Exception):
    pass
try:
    raise CustomError("This is a custom error message.")
except CustomError as e:
    print("Caught a custom exception:", e)

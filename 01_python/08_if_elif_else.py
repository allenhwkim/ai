"""
Conditional statements in Python allow you to execute code based on certain conditions. 
The main conditional statements are `if`, `elif`, and `else`.
"""

# 1. if, elif, and else
x = 10
if x < 0:
    print("x is negative")
elif x == 0:
    print("x is zero")
else:
    print("x is positive")

# 2. Nested if statements
y = 5
if y > 0:
    print("y is positive")
    if y % 2 == 0:
        print("y is even")
    else:
        print("y is odd")

# 3. Logical operators in if statements
a = True
b = False
if a and b:
    print("Both a and b are True")
elif a or b:
    print("At least one of a or b is True")
else:
    print("Neither a nor b is True")

# 4. comparison operators in if statements
num = 15
if num < 10:
    print("num is less than 10")
elif 10 <= num < 20:
    print("num is between 10 and 20")
else:
    print("num is 20 or greater")

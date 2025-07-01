"""
While loops in Python are used to repeatedly execute a block of code as long as a specified condition is true.
They are particularly useful when the number of iterations is not known beforehand.
"""

# 1. Basic while loop
count = 0
while count < 5:
    print("Count is:", count)
    count += 1  # Increment count to avoid infinite loop
# Output: Count is: 0, Count is: 1, Count is: 2, Count is: 3, Count is: 4

# 2. Using break to exit the loop
count = 0
while True:  # Infinite loop
    if count >= 5:
        break
    print("Count is:", count)
    count += 1  # Increment count to avoid infinite loop
# Output: Count is: 0, Count is: 1, Count is: 2, Count is: 3, Count is: 4

# 3. Using continue to skip an iteration
count = 0
while count < 5:
    count += 1  # Increment count first
    if count == 3:
        continue  # Skip the rest of the loop when count is 3
    print("Count is:", count)
# Output: Count is: 1, Count is: 2, Count is: 4, Count is: 5  

# 4. Using else with while loop
count = 0
while count < 5:
    print("Count is:", count)
    count += 1  # Increment count to avoid infinite loop
else:
    print("Count has reached 5, exiting loop.")
# Output: Count is: 0, Count is: 1, Count is: 2, Count is: 3, Count is: 4, Count has reached 5, exiting

# 5. Using a while loop with user input
while True:
    user_input = input("Enter a number (or 'exit' to quit): ")
    if user_input.lower() == 'exit':
        print("Exiting the loop.")
        break  # Exit the loop if user types 'exit'
    try:
        number = int(user_input)
        print("You entered:", number)
    except ValueError:
        print("That's not a valid number. Please try again.")
# Output will depend on user input, e.g.:
# Enter a number (or 'exit' to quit): 5
# You entered: 5
# Enter a number (or 'exit' to quit): exit
# Exiting the loop.

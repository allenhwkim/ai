# 1. define strings
a = "hello"
b = 'hello'
c = """hello"""
print(a, b, c) # hello hello hello

# 2. string methods
print('a.upper():', a.upper()) # HELLO
print('a.lower():', a.lower()) # hello
print('a.capitalize():', a.capitalize()) # Hello
print('a.title():', a.title()) # Hello
print('a.strip():', a.strip()) # hello
print('a.strip("h"):', a.strip('h')) # ello
print('a.strip("o"):', a.strip('o')) # hell
print('a.replace("l", "x"):', a.replace("l", "x")) # hexxo
print('a.split(""):', a.split("")) # ['', 'h', 'e', 'l', 'l', 'o', ''] # splits into characters
print('a.split("l"):', a.split("l")) # ['he', '', 'lo']
print('a.find("l"):', a.find("l")) # 2
print('a.index("l"):', a.index("l")) # 2
print('a.count("l"):', a.count("l")) # 2
print('a.startswith("he"):', a.startswith("he")) # True
print('a.endswith("lo"):', a.endswith("lo")) # True
print('a.isalpha():', a.isalpha()) # True
print('a.isdigit():', a.isdigit()) # False
print('a.isalnum():', a.isalnum()) # True
print('a.islower():', a.islower()) # True
print('a.isupper():', a.isupper()) # False
print('a.isspace():', a.isspace()) # False
print('a.isnumeric():', a.isnumeric()) # False
print('a.isascii():', a.isascii()) # True
print('a.isprintable():', a.isprintable()) # True
print('a.swapcase():', a.swapcase()) # HELLO
print('a.center(10):', a.center(10)) # '  hello   '
print('a.ljust(10):', a.ljust(10)) # 'hello     '
print('a.rjust(10):', a.rjust(10)) # '     hello'
print('a.zfill(10):', a.zfill(10)) # '00000hello'

# 3. string formatting
name = "Allen"
age = 30
print(f"My name is {name} and I am {age} years old.") # My name is Allen and I am 30 years old.

# 4. string concatenation
str1 = "Hello"
str2 = "World"
print(str1 + " " + str2) # Hello World

# 5. string repetition
print(str1 * 3) # HelloHelloHello

# 6. string slicing
s = "Hello, World!"
print(s[0:5]) # Hello
print(s[7:]) # World!
print(s[:5]) # Hello
print(s[-6:]) # World!
print(s[::2]) # Hlo ol!
print(s[::-1]) # !dlroW ,olleH

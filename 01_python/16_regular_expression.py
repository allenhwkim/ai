"""
Regular expressions (regex) are a powerful tool for matching patterns in strings.
They can be used for searching, replacing, and validating strings 
based on specific patterns.
"""

import re

# 1. Basic pattern matching
text = "Hello, World!"
match = re.search(r"Hello", text)
if match:
    print(f"Match found: {match.group()}")
else:
    print("No match found.")
    
# 2. Matching digits
text = "There are 123 apples."
matches = re.findall(r"\d+", text)
print(f"Digits found: {matches}")

# 3. Matching words
text = "Python is great, and Python is versatile."
matches = re.findall(r"\bPython\b", text)
print(f"Word 'Python' found {len(matches)} times.")

# 4. Replacing text
text = "I love Python programming."
new_text = re.sub(r"Python", "JavaScript", text)
print(f"Replaced text: {new_text}")

# 5. Splitting text
text = "apple, banana, cherry"
fruits = re.split(r",\s*", text)
print(f"Fruits list: {fruits}")

# 6. Patterns
email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}" # me@email.com
phone_pattern = r"\d{3}-\d{3}-\d{4}" # 123-456-7890
url_pattern = r"https?://[a-zA-Z0-9.-]+(?:/[a-zA-Z0-9._%+-]*)*" # https://www.example.com/path/to/resource
date_pattern = r"\d{4}-\d{2}-\d{2}"  # 2025-06-30
time_pattern = r"\d{2}:\d{2}(:\d{2})?"  # 14:30 or 14:30:59
ipv4_pattern = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"  # 192.168.1.1
hex_color_pattern = r"#(?:[0-9a-fA-F]{3}){1,2}\b"  # #fff or #ffffff
postal_code_pattern = r"\b\d{5}(?:-\d{4})?\b"  # 12345 or 12345-6789
username_pattern = r"@[A-Za-z0-9_]+"  # @username
hashtag_pattern = r"#[A-Za-z0-9_]+"  # #hashtag
float_pattern = r"-?\d+\.\d+"  # 3.14, -0.001





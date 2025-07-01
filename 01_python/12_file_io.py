"""
File I/O in Python allows you to read from and write to files on your system. 
This is essential for data persistence, allowing you to save your program's state or results for later use.
You can work with text files, binary files, and even structured data formats like JSON or CSV.
"""

# 1. Writing to a file
with open('example.txt', 'w') as file:
    file.write("Hello, World!\n")
    file.write("This is a file I/O example.\n")

# 2. Reading from a file
with open('example.txt', 'r') as file:
    content = file.read()
    print("File content:")
    print(content) # Displays the content of the file

# 3. Appending to a file
with open('example.txt', 'a') as file:
    file.write("Appending a new line.\n")

# 4. Reading lines from a file
with open('example.txt', 'r') as file:
    lines = file.readlines()
    print("File lines:")
    for line in lines:
        print(line.strip())  # Strip newline characters

# 5. Reading a file line by line
with open('example.txt', 'r') as file:
    for line in file:
        print(line.strip())  # Process each line individually 

# 6. Working with JSON files
import json
data = {
    "name": "Alice",
    "age": 30,
    "city": "New York"
}
with open('data.json', 'w') as file:
    json.dump(data, file, indent=4)  # Write JSON data with indentation for readability
with open('data.json', 'r') as file:
    json_data = json.load(file)  # Load JSON data from the file
    print("JSON data:", json_data)  # Displays the JSON data read from the file 


# 7. Working with CSV files
import csv
with open('data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Name', 'Age', 'City'])  # Write header
    writer.writerow(['Alice', 30, 'New York'])  # Write a row of data
    writer.writerow(['Bob', 25, 'Los Angeles'])  # Write another row of data
with open('data.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        print("CSV row:", row)  # Displays each row of the CSV file 

# 8. Handling file exceptions
try:
    with open('non_existent_file.txt', 'r') as file:
        content = file.read()
except FileNotFoundError:
    print("Error: The file does not exist.")

# 9. File existence check
import os
if os.path.exists('example.txt'):
    print("The file 'example.txt' exists.")
else:
    print("The file 'example.txt' does not exist.")

# 10. Listing files in a directory
directory = '.'  # Current directory
files = os.listdir(directory)  # List all files in the directory
print("Files in directory:", files)  # Displays the list of files in the current directory  

# 11. Checking if a path is a file or directory
for file in files:
    file_path = os.path.join(directory, file)  # Get the full path of the file
    if os.path.isfile(file_path):
        print(f"{file} is a file.")
    elif os.path.isdir(file_path):
        print(f"{file} is a directory.")
    else:
        print(f"{file} is neither a file nor a directory.")  # Should not happen in normal cases  

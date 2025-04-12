import re
import sys
import os

filename = sys.argv[1]

if not os.path.isfile(filename):
    print(f"Error: The file '{filename}' does not exist.")
    sys.exit()

def clean_text(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        book_text = file.read()

    cleaned_text = re.sub(r'\n+', ' ', book_text)  # replace new lines
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # replace multiple spaces

    cleanFileName = filename + ".cleaned"
    print(cleanFileName, len(cleaned_text), "characters")  # num of characters

    with open(cleanFileName, 'w', encoding='utf-8') as file:
        file.write(cleaned_text)

clean_text(filename)


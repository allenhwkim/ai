import pandas as pd
import numpy as np

data = {
    'Names': ['Gulshan', 'Shashank', 'Bablu', 'Abhishek', 'Anand', np.nan, 'Pratap'],
    'City': ['Delhi', 'Mumbai', 'Kolkata', 'Delhi', 'Chennai', 'Bangalore', 'Hyderabad']
}
df = pd.DataFrame(data)
print("Data types before conversion:\n", df.dtypes)

# Convert columns to string types
print('str', str) # str <class 'str'>
df['Names'] = df['Names'].astype(str)  # Convert 'Names' column to string type
df['City'] = df['City'].astype(str)  # Convert 'City' column to string type
print("Data types after conversion:\n", df.dtypes)


# 1. Convert a column to lowercase and uppercase
# .str is a string accessor that allows vectorized string operations on Series
df['Names'] = df['Names'].str.lower()  # Convert 'Names' column to lowercase 
df['City'] = df['City'].str.upper()  # Convert 'City' column to uppercase
print("Names column after converting to lowercase:\n", df)


# 2. Convert a column to title case
df = pd.DataFrame(data)
df['Names'] = df['Names'].str.title()  # Convert 'Names' column to title case
print("Names column after converting to title case:\n", df)

# 3. Replace substrings in a column
df = pd.DataFrame(data)
df = pd.DataFrame(data)  # Reset DataFrame for demonstration
df['City'] = df['City'].str.replace('Delhi', 'New Delhi')  # Replace 'Delhi' with 'New Delhi' in 'City' column
print("City column after replacing 'Delhi' with 'New Delhi':\n", df)

# 4. Check if a substring exists in a column
df = pd.DataFrame(data)
df['Has_Gulshan'] = df['Names'].str.contains('Gulshan')
print("DataFrame after checking if 'Gulshan' exists in 'Names':\n", df)

# 5. Split a column into multiple columns
df = pd.DataFrame(data)  # Reset DataFrame for demonstration
last_names = ['Gulshan', 'Shashank', 'Bablu', 'Abhishek', 'Anand', 'Pratap', 'Singh']
df['Names'] = df['Names'].fillna('')
print("DataFrame after filling NaN with last names:\n", df)

df['Names'] = df['Names'] + ' ' + pd.Series(last_names)  # Fill NaN and concatenate last names
print("DataFrame after filling NaN and concatenating last names:\n", df)
# n=1 means split at the first occurrence of space, expand=True creates separate columns
df[['First_Name', 'Last_Name']] = df['Names'].str.split(' ', n=1, expand=True) 
print("DataFrame after splitting 'Names':\n", df)

# 6. Concatenate two columns into one
df['Full_Name'] = df['First_Name'] + ' ' + df['Last_Name']  # Concatenate 'First_Name' and 'Last_Name' into 'Full_Name'
print("DataFrame after concatenating 'First_Name' and 'Last_Name' into 'Full_Name':\n", df)

# 7. Count occurrences of a substring in a column
df = pd.DataFrame(data)
count = df['Names'].str.count('Gulshan')
print("DataFrame after counting occurrences of 'Gulshan' in 'Names':\n", count)

# 8. Remove special characters from a column
df = pd.DataFrame(data)
df['Names'] = '#' + df['Names']
df['Removed'] = df['Names'].str.replace(r'[^a-zA-Z\s]', '', regex=True)  # Remove special characters from 'Names'
print("DataFrame after removing special characters from 'Names':\n", df[['Names', 'Removed']])  

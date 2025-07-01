"""
In Pandas, missing values are represented as:
  - None: A Python object used to represent missing values in object-type arrays.
  - NaN: Not a bumber, A special floating-point value.
"""

import pandas as pd
import numpy as np

# 1. Find columns with null values in a DataFrame
d = {
    'First Score': [100, 90, np.nan, 95],
    'Second Score': [30, 45, 56, np.nan],
    'Third Score': [np.nan, 40, 80, 98]
}
df = pd.DataFrame(d)
mv = df.isnull()
print(mv)

cols_with_nulls = df.columns[df.isnull().any()]
print("Columns with null values:", cols_with_nulls)

# 2. Print out the rows with missing values
df = pd.read_csv("data/employees.csv")
cols_with_nulls = df.columns[df.isnull().any()]
print("Columns with missing values:", cols_with_nulls)

bool_series = pd.isnull(df["Gender"]) # 
print(bool_series) # 0 False 1 False 2 True 3 False 4 False ...
print(type(bool_series)) # dtype: <class 'pandas.core.series.Series'>
print("Rows with missing values in Gender column:", df[bool_series])

# 3. Fill missing values with a specific value
df = pd.read_csv("data/employees.csv")
df['Gender'].fillna("Unknown", inplace=True)  # Fill all NaN values with "Unknown"
print("Data after filling missing values in 'Gender' column:\n", df)

# 4. Fill missing values with 0
d = {
    'First Score': [100, 90, np.nan, 95],
    'Second Score': [30, 45, 56, np.nan],
    'Third Score': [np.nan, 40, 80, 98]
}
df = pd.DataFrame(d)
df.fillna(0, inplace=True)  # Fill all NaN values with 0
print("DataFrame after filling NaN with 0:\n", df)

# 5. Fill missing values with certain values
df = pd.read_csv("data/employees.csv")
print("Data before filling missing values:\n", df[20: 25])
df['Gender'].fillna('No Gender', inplace=True)  # Fill NaN in 'Genger' column with)
print("Data after filling missing values in 'Genger' column:\n", df[20: 25])

# 6. replace all NaN values with a specific value
df = pd.read_csv("data/employees.csv")
print("Data before replacing NaN values:\n", df[20: 25])
df.replace(np.nan, -99, inplace=True)  # Replace all NaN values to -99
print("Data after replacing NaN values with -99:\n", df[20: 25])

# 7. Drop rows with missing values
df = pd.read_csv("data/employees.csv")
# drop all rows with any gender NaN values
df2 = df.dropna(subset=['Gender'])
print("Rows before dropping NaN values:", len(df))
print("Rows after dropping NaN values:", len(df2))

# 8. Drop duplicate rows
df = pd.read_csv("data/employees.csv")
df2 = df.drop_duplicates(subset=['First Name','Gender'], keep='first')  # Keep the first occurrence of each duplicate
print("Rows before dropping NaN values:", len(df))
print("Rows after dropping NaN values:", len(df2))

# 9. Drop empty columns
data = {
    'FirstName': ['Vipul', 'Ashish', 'Milan'],
    "Gender": ["", "", ""],
    "Age": [0, 0, 0],
    "Department": [None, None, None],
} 
df = pd.DataFrame(data)
print("Data before dropping empty columns:\n", df)
# Drop columns where all values are NaN
df.dropna(axis=1, how='all', inplace=True) 
print("Data after dropping empty columns:\n", df)
# Drop blank columns
df = df.replace('', np.nan)  # Replace empty strings with NaN
df.dropna(axis=1, how='all', inplace=True)  # Drop columns where all values are NaN
print("Data after dropping blank columns:\n", df)
# Drop columns with all zero values
df = df.replace(0, np.nan)  # Replace 0 with NaN
df.dropna(axis=1, how='all', inplace=True)  # Drop columns where all values are NaN
print("Data after dropping columns with all zero values:\n", df)


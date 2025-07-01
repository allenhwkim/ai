"""
Pandas offers a variety of functions to read data from and write data to different file formats 
"""

import pandas as pd

# 1. Reading from CSV files
df_csv = pd.read_csv('data/people_data.csv')  # Reads a CSV file into a DataFrame
print("DataFrame from CSV:\n", df_csv)

# 2. Write/Reading from .json files
df_csv.to_json('data/people_data.json', orient='records', lines=True)  # Writes DataFrame to JSON file
df_csv = pd.read_json('data/people_data.json', lines=True)  # Reads a JSON file into a DataFrame
print("\nDataFrame from JSON:\n", df_csv)

# 3. Write/Reading from Excel files
df_csv.to_excel('data/people_data.xlsx', sheet_name='Sheet1', index=False)  # Writes DataFrame to Excel file
df_csv = pd.read_excel('data/people_data.xlsx', sheet_name='Sheet1')  # Reads an Excel file into a DataFrame
print("\nDataFrame from Excel:\n", df_csv)

# 4. Read with specific options
# Reading only specific columns from a CSV file
df_csv = pd.read_csv('data/people_data.csv', usecols=["First Name", "Last Name"]) 
print("\nDataFrame from CSV with selected columns:\n", df_csv)

# Reading a CSV file with a specific index column
df_csv = pd.read_csv('data/people_data.csv', index_col="First Name")
print("\nDataFrame from CSV with 'First Name' as index:\n", df_csv)

# Reading a CSV file with limited rows
df_csv = pd.read_csv('data/people_data.csv', nrows=3)  # Reads only the first 3 rows
print("\nDataFrame from CSV with limited rows:\n", df_csv)

# Reading with date parsing
df_csv = pd.read_csv('data/people_data.csv', parse_dates=["Date of birth"])  # Parses 'Date of Birth' as datetime
print("\nDataFrame from CSV with parsed dates:\n", df_csv)
print("\nColumn types:\n", df_csv.dtypes)

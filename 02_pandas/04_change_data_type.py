"""
To change the data type of a column in a Pandas DataFrame,
<column>.astype() method is used to convert a specific column 
to a desired data type.
"""
import pandas as pd

# 1. Change data type of a column in a DataFrame
data = {
    'Name': ['John', 'Alice', 'Bob', 'Eve', 'Charlie'], 
    'Age': [25, 30, 22, 35, 28], 
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'], 
    'Salary': [50000, 55000, 40000, 70000, 48000]
}
df = pd.DataFrame(data)
print("Original data types:\n", df.dtypes) # Print original data types
df['Age'] = df['Age'].astype(float)
print("Data types after change:\n", df.dtypes)  # Print data types after change

# 2. Change data type of multiple columns in a DataFrame
df = pd.DataFrame(data)
df = df.astype({'Age': float, 'Salary': float})  # Change multiple columns to float using astype
print("Data types after changing multiple columns using astype:\n", df.dtypes)

# 3. Converting a Column to a DateTime Type
df['Joining Date'] = ['2020-01-01', '2019-05-15', '2021-03-20', '2018-07-30', '2022-11-10']
# df['Joining Date'] = pd.to_datetime(df['Joining Date'])  # Convert to datetime
df = df.astype({'Joining Date': 'datetime64[ns]'})  # Ensure the column is of datetime type
print("Data types after converting 'Joining Date' to datetime:\n", df.dtypes)



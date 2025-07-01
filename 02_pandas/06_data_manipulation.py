import pandas as pd

df = pd.DataFrame()
df['Name'] = ['Abhijit','Smriti', 'Akash', 'Roshni']
df['Age'] = [20, 19, 20, 14]
df['Student'] = [False, True, True, False]
print(df)
print("\ndescribe:\n", df.describe())  # Output: Summary statistics for numeric columns
print("\ndescribe:\n", df.describe(include=['bool', 'number'])) 
print("\nshape:\n", df.shape)  # Output: (4, 3)
print("\ninfo:\n ", df.info())  # Output: <class 'pandas.core.frame.DataFrame'> RangeIndex: 4 entries, 0 to 3 Data columns (total 3 columns): #   Column   Non-Null Count  Dtype  --- ------   --------------  ----- 0   Name     4 non-null      object 1   Age      4 non-null      int64 2   Student  4 non-null      bool dtypes: bool(1), int64(1), object(1) memory usage: 200.0+ bytes
print("\ncorrelation:\n", df.corr(numeric_only=True))  # Output: Correlation matrix for numeric columns

# 1. Add a new row to the DataFrame
new_row = {'Name': 'Rahul', 'Age': 21, 'Student': True}
df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
print("\nDataFrame after adding a new row:\n", df)

# 2. Add a new column to the DataFrame
df = df.assign(Grade=['A', 'B', 'C', 'A', 'F'])
print("\nDataFrame after adding 'Grade' column:\n", df) 

# 3. Rename columns in the DataFrame
df.rename(columns={'Name': 'First_Name', 'Age': 'Years', 'Student': 'Is_Student'}, inplace=True)
print("\nDataFrame after renaming columns:\n", df)

# 4. Drop a column from the DataFrame
df.drop(columns=['Grade'], inplace=True)
print("\nDataFrame after dropping 'Grade' column:\n", df) 

# 5. Drop a row from the DataFrame
df.drop(index=0, inplace=True)  # Drop the first row
print("\nDataFrame after dropping the first row:\n", df)

# 6. Filter rows based on a condition
filtered_df = df[df['Years'] > 18]  # Filter rows where 'Years' is greater than 18
print("\nFiltered DataFrame (Years > 18):\n", filtered_df)

"""
When working with tabular data, such as data stored in spreadsheets 
or databases pandas is the right tool for you. pandas will help you
to explore, clean, and process your data. In pandas, a data table 
is called a DataFrame.
"""

"""
Pandas provides two primary data structures:
1. Series: A one-dimensional labeled array capable of holding any data type.
2. DataFrame: A two-dimensional labeled data structure with columns of potentially different types, similar
"""

import pandas as pd

# Create a Series
s = pd.Series([1, 2, 3])
print("Series:", s)
# Series: 0    1
# 1    2
# 2    3
# dtype: int64

# Create a DataFrame
data = {
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9]
}
df = pd.DataFrame(data)
print("DataFrame:\n", df)
# DataFrame:
#    A  B  C
# 0  1  4  7
# 1  2  5  8
# 2  3  6  9 

# Accessing DataFrame columns
print("Column A:\n", df['A'])
# Column A:
# 0    1
# 1    2
# 2    3
# Name: A, dtype: int64

# Accessing DataFrame rows by index
print("Row 0:\n", df.iloc[0])
# Row 0:
# A    1
# B    4
# C    7
# Name: 0, dtype: int64


# Accessing DataFrame rows by label
print("Row with index 1:\n", df.loc[1])
# Row with index 1:
# A    2
# B    5
# C    8
# Name: 1, dtype: int64


# Adding a new column to the DataFrame
df['D'] = df['A'] + df['B']
print("DataFrame after adding column D:\n", df)
# DataFrame after adding column D:
#    A  B  C   D
# 0  1  4  7   5
# 1  2  5  8   7
# 2  3  6  9   9 

# Removing a column from the DataFrame
df.drop('D', axis=1, inplace=True)
print("DataFrame after removing column D:\n", df)
# DataFrame after removing column D:
#    A  B  C
# 0  1  4  7
# 1  2  5  8
# 2  3  6  9


# Filtering rows based on a condition
filtered_df = df[df['A'] > 1]
print("Filtered DataFrame (A > 1):\n", filtered_df)
# Filtered DataFrame (A > 1): 
#    A  B  C
# 1  2  5  8
# 2  3  6  9  


# Grouping data by a column and calculating the mean
grouped_df = df.groupby('A').mean()
print("Grouped DataFrame by A with mean:\n", grouped_df)
# Grouped DataFrame by A with mean:
#      B    C
# A
# 1  4.0  7.0
# 2  5.0  8.0
# 3  6.0  9.0

# Merging two DataFrames
data2 = {
    'A': [1, 2, 3],
    'E': [10, 20, 30] # New column to merge on
}
df2 = pd.DataFrame(data2)
merged_df = pd.merge(df, df2, on='A', how='inner')
print("Merged DataFrame:\n", merged_df)
# Merged DataFrame:
#    A  B  C   E
# 0  1  4  7  10
# 1  2  5  8  20
# 2  3  6  9  30

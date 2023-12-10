import pandas as pd

# Creating a DataFrame
data = {'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Emily'],
        'Age': [25, 30, 35, 22, 28],
        'City': ['New York', 'San Francisco', 'Los Angeles', 'Chicago', 'Boston']}
df = pd.DataFrame(data)

# Displaying basic information about the DataFrame
print("DataFrame:")
print(df)
print("\nDataFrame Info:")
print(df.info())

# Descriptive statistics
print("\nDescriptive Statistics:")
print(df.describe())

# Selecting columns
print("\nSelecting Columns:")
print(df['Name'])
print(df[['Name', 'Age']])

# Filtering rows
print("\nFiltering Rows:")
filtered_df = df[df['Age'] > 25]
print(filtered_df)

# Sorting
print("\nSorting DataFrame:")
sorted_df = df.sort_values(by='Age', ascending=False)
print(sorted_df)

# Adding a new column
df['Senior'] = df['Age'] > 30
print("\nDataFrame with New Column:")
print(df)

# Grouping and Aggregation
grouped_df = df.groupby('City').mean()
print("\nGrouped DataFrame:")
print(grouped_df)

# Handling missing data
df.loc[2, 'Age'] = None
print("\nDataFrame with Missing Data:")
print(df)
print("\nDataFrame after Handling Missing Data:")
df.dropna(inplace=True)
print(df)

# Reading from and writing to files
df.to_csv('example.csv', index=False)
read_df = pd.read_csv('example.csv')
print("\nRead DataFrame from CSV:")
print(read_df)

# Applying functions to columns
df['Name_Length'] = df['Name'].apply(len)
print("\nDataFrame with New Column (Name Length):")
print(df)

# Merging DataFrames
df2 = pd.DataFrame({'Name': ['Bob', 'Charlie', 'David'], 'Salary': [60000, 80000, 70000]})
merged_df = pd.merge(df, df2, on='Name')
print("\nMerged DataFrame:")
print(merged_df)
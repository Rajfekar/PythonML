import numpy as np

# Creating NumPy arrays
array1 = np.array([1, 2, 3, 4, 5])
array2 = np.arange(1, 10, 2)  # Create an array with values from 1 to 10 (exclusive) with step 2
array3 = np.linspace(0, 1, 5)  # Create an array with 5 equally spaced values between 0 and 1

# Basic array operations
sum_array = array1 + array2
product_array = array1 * array2
mean_value = np.mean(array1)

# Reshaping arrays
reshaped_array = array1.reshape(5, 1)

# Indexing and slicing
subset = array1[1:4]  # Extract elements from index 1 to 3

# Array functions
max_value = np.max(array1)
min_value = np.min(array2)

# Random number generation
random_array = np.random.rand(3, 3)  # Create a 3x3 array with random values between 0 and 1

# Linear algebra operations
matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6], [7, 8]])
matrix_product = np.dot(matrix1, matrix2)

# Statistical operations
mean_value = np.mean(array1)
standard_deviation = np.std(array2)

# Universal functions (ufunc)
squared_array = np.square(array1)
sin_values = np.sin(array3)

# Boolean indexing
boolean_indexing = array1[array1 > 3]

# Stacking arrays
stacked_array = np.vstack((array1, array2))

# Iterating through arrays
for element in array1:
    print(element)

# Printing arrays
print("Array 1:", array1)
print("Array 2:", array2)
print("Array 3:", array3)
print("Sum of arrays:", sum_array)
print("Product of arrays:", product_array)
print("Reshaped array:", reshaped_array)
print("Subset of array:", subset)
print("Max value:", max_value)
print("Min value:", min_value)
print("Random array:", random_array)
print("Matrix product:", matrix_product)
print("Mean value:", mean_value)
print("Standard deviation:", standard_deviation)
print("Squared array:", squared_array)
print("Sin values:", sin_values)
print("Boolean indexing:", boolean_indexing)
print("Stacked array:", stacked_array)
# Broadcasting
broadcasted_array = array1 + 10  # Broadcasting scalar value to each element in the array

# Aggregation functions
sum_all_elements = np.sum(array1)
cumulative_sum = np.cumsum(array1)

# Transposing arrays
transposed_matrix = matrix1.T

# Concatenation
concatenated_array = np.concatenate((array1, array2))

# Splitting arrays
split_arrays = np.split(array1, [2, 4])  # Split at indices 2 and 4

# Sorting arrays
sorted_array = np.sort(array1)

# Unique elements
unique_elements = np.unique(array1)

# Saving and loading arrays
np.save('saved_array.npy', array1)
loaded_array = np.load('saved_array.npy')

# Example of using np.where
where_condition = np.where(array1 > 3, "Greater", "Less or Equal")

# Example of using np.ma.masked_array to mask certain values
masked_array = np.ma.masked_array(array1, mask=array1 < 3)

# Example of using np.meshgrid
x = np.arange(-5, 5, 1)
y = np.arange(-5, 5, 1)
xx, yy = np.meshgrid(x, y)
z = xx**2 + yy**2  # Example equation for a 2D grid

# Printing additional results
print("Broadcasted array:", broadcasted_array)
print("Sum of all elements:", sum_all_elements)
print("Cumulative sum:", cumulative_sum)
print("Transposed matrix:", transposed_matrix)
print("Concatenated array:", concatenated_array)
print("Split arrays:", split_arrays)
print("Sorted array:", sorted_array)
print("Unique elements:", unique_elements)
print("Loaded array:", loaded_array)
print("Where condition result:", where_condition)
print("Masked array:", masked_array)
print("Meshgrid example:")
print("X values:\n", xx)
print("Y values:\n", yy)
print("Z values (example equation):\n", z)

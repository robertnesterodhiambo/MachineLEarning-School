# Variable declaration
boolean_value = True          # Boolean
integer_value = 42           # Numbers
float_value = 3.14           # Floating-point number
string_value = "Hello, World!"  # Strings

# Log statement
print("Variable Declaration Complete!")

# Type check
print(type(boolean_value))   # Output: <class 'bool'>
print(type(integer_value))    # Output: <class 'int'>
print(type(float_value))      # Output: <class 'float'>
print(type(string_value))     # Output: <class 'str'>

# Length check
print(len(string_value))      # Output: 13

# Comments
# This is a single-line comment

"""
This is a 
multiline comment
"""

# Data Types

# Primitive Data Types
# Boolean
is_active = False

# Numbers
age = 30
height = 5.9

# Strings
name = "John Doe"

# Composite Data Types

# List
# Initialize
fruits = ["apple", "banana", "cherry"]
# Access value
print(fruits[1])             # Output: banana
# Change value
fruits[0] = "orange"
print(fruits)                # Output: ['orange', 'banana', 'cherry']
# Add value
fruits.append("grape")
print(fruits)                # Output: ['orange', 'banana', 'cherry', 'grape']
# Delete value
fruits.remove("banana")
print(fruits)                # Output: ['orange', 'cherry', 'grape']

# Tuples
# Initialize
colors = ("red", "green", "blue")
# Access value
print(colors[1])             # Output: green
# Change value (will raise an error)
# colors[0] = "yellow"      # Uncommenting this will raise a TypeError
# Add value (will raise an error)
# colors.append("yellow")    # Uncommenting this will raise an AttributeError
# Delete value (will raise an error)
# del colors[1]              # Uncommenting this will raise a TypeError

# Dictionary
# Initialize
person = {
    "name": "Alice",
    "age": 25,
    "favorite_team": "Warriors"
}
# Access value
print(person["name"])        # Output: Alice
# Change value
person["age"] = 26
print(person)                # Output: {'name': 'Alice', 'age': 26, 'favorite_team': 'Warriors'}
# Add value
person["city"] = "New York"
print(person)                # Output: {'name': 'Alice', 'age': 26, 'favorite_team': 'Warriors', 'city': 'New York'}
# Delete value
del person["favorite_team"]
print(person)                # Output: {'name': 'Alice', 'age': 26, 'city': 'New York'}

# Conditional Statements
# If
if age < 18:
    print("Minor")
# Else If
elif age >= 18 and age < 65:
    print("Adult")
# Else
else:
    print("Senior")

# For Loop
# Start: 0, Stop: 5, Increment: 1
for i in range(5):
    print(i)                  # Output: 0, 1, 2, 3, 4
    # Break example
    if i == 3:
        break                 # Stops the loop when i equals 3
# Continue example
for j in range(5):
    if j == 2:
        continue              # Skips the iteration when j equals 2
    print(j)                  # Output: 0, 1, 3, 4

# While Loop
# Start: 0, Stop: 5, Increment: 1
count = 0
while count < 5:
    print(count)             # Output: 0, 1, 2, 3, 4
    count += 1              # Increment

# Function Definition
def greet(name):             # Parameter
    return f"Hello, {name}!"  # Return statement

# Function Call
greeting = greet("Alice")    # Argument
print(greeting)               # Output: Hello, Alice!

# Bonus: Errors
try:
    print(undeclared_variable)  # NameError: name 'undeclared_variable' is not defined
except NameError as e:
    print(e)

try:
    tuple_example = (1, 2, 3)
    tuple_example[0] = 10     # TypeError: 'tuple' object does not support item assignment
except TypeError as e:
    print(e)

try:
    my_dict = {"name": "Alice"}
    print(my_dict["favorite_team"])  # KeyError: 'favorite_team'
except KeyError as e:
    print(e)

try:
    my_list = [1, 2, 3]
    print(my_list[5])         # IndexError: list index out of range
except IndexError as e:
    print(e)

try:
    def sample_function():
        print("Hello")
        print("World")   # IndentationError: unexpected indent
        return
except IndentationError as e:
    print(e)

try:
    my_tuple = (1, 2, 3)
    my_tuple.pop()          # AttributeError: 'tuple' object has no attribute 'pop'
except AttributeError as e:
    print(e)

try:
    my_tuple = (1, 2, 3)
    my_tuple.append(4)      # AttributeError: 'tuple' object has no attribute 'append'
except AttributeError as e:
    print(e)

# Tuples - Attempt to change, add, or delete
try:
    my_tuple = (1, 2, 3)
    my_tuple[0] = 10        # Change value
except TypeError as e:
    print(e)

try:
    my_tuple += (4,)        # Add value (this is technically allowed because it creates a new tuple)
    print(my_tuple)         # Output: (1, 2, 3, 4)
except TypeError as e:
    print(e)

try:
    del my_tuple[1]        # Delete value
except TypeError as e:
    print(e)


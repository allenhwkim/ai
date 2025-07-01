# 1. Numeric types 
a = 1
print(a, type(a)) # 1 <class 'int'>

float = 1.23
print(float, type(float)) # 1.23 <class 'float'>

complex = 10 + 5j
print("complex real part:", complex.real) # 10.0
print("complex imaginary part:", complex.imag) # 5.0
print("complex conjugate:", complex.conjugate()) # (10-5j)
print(complex, type(complex)) # (10+5j) <class 'complex'>

# 2. Sequence types 
str = "Hello"
print(str, type(str)) # "Hello" <class 'str'>

list = [1, 2, 3]
print(list, type(list)) # [1, 2, 3] <class 'list'>

tuple = (1, 2, 3)
print(tuple, type(tuple)) # (1, 2, 3) <class 'tuple'>

range = range(5)
print(range, type(range)) # range(0, 5) <class 'range'>

# 3. Dictionary type
dict = {"key1": "value1", "key2": "value2"}
print(dict, type(dict)) # {'key1': 'value1', 'key2': 'value2'} <class 'dict'>

# 4. Set types
set = {1, 1, 1, 2, 3}
print(set, type(set)) # {1, 2, 3} <class 'set'>

frozenset = frozenset([1, 2, 3])
print(frozenset, type(frozenset)) # frozenset({1, 2, 3}) <class 'frozenset'>

# 5. Boolean type
bool_true = True
bool_false = False
print(bool_true, type(bool_true)) # True <class 'bool'>
print(bool_false, type(bool_false)) # False <class 'bool'>

# 6. None type
none = None
print(none, type(none)) # None <class 'NoneType'> 

# 7. Bytes and bytearray types
bytes_data = b"Hello"
print(bytes_data, type(bytes_data)) # b'Hello' <class 'bytes'>  

bytearray_data = bytearray(b"Hello")
print(bytearray_data, type(bytearray_data)) # bytearray(b'Hello') <class 'bytearray'>

# 8. Memoryview type
data = bytearray(b'abc')
memoryview_data = memoryview(bytes_data)
print(memoryview_data, type(memoryview_data)) # <memory at 0x...> <class 'memoryview'> 


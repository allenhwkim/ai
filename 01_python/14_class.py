"""
Classes in Python are blueprints for creating objects. 
They encapsulate data for the object and methods to manipulate 
that data.
"""

# 1. Define a class
class Dog:
    def __init__(self, name, age):
        self.name = name  # Instance variable
        self.age = age    # Instance variable

    def bark(self):
        return f"{self.name} says Woof!"

    def get_age(self):
        return self.age

dog = Dog("Buddy", 3)
print(dog.bark())  # Buddy says Woof!
print(f"{dog.name} is {dog.get_age()} years old.")  # Buddy is 3 years old.

# 2. Class with class variables
class Cat:
    species = "Felis catus"  # Class variable

    def __init__(self, name):
        self.name = name  # Instance variable

    def meow(self):
        return f"{self.name} says Meow!"

cat = Cat("Whiskers")
print(cat.meow())  # Whiskers says Meow!
print(f"{cat.name} belongs to the species {Cat.species}.")  # Whiskers belongs to the species Felis catus.

# 3. Inheritance
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        return f"{self.name} makes a sound."  # Base class

class Dog(Animal):  # Inherits from Animal
    def speak(self):
        return f"{self.name} barks."  # Overriding the speak method
    
class Cat(Animal):  # Inherits from Animal
    def speak(self):
        return f"{self.name} meows."  
    
dog = Dog("Rex")
cat = Cat("Mittens")
print(dog.speak())  # Rex barks.
print(cat.speak())  # Mittens meows.

# 4. Class attributes and properties
# Class attributes are shared across all instances of the class.
# Properties allow controlled access to instance attributes.
# Private attribute starts with _ or __
# Attribute starting with _ can be accessible. attribute starting with __ cannot be accessible.

class Person:
    def __init__(self, name, age):
        self._name = name  # Protected attribute
        self.__age = age   # Private attribute

    @property # class property, define getter, setter and deleter for name
    def name(self):
        return self._name

    @property
    def age(self):
        return self.__age

    @age.setter # extend the age property to control age setting
    def age(self, value):
        if value < 0:
            raise ValueError("Age cannot be negative.")
        self.__age = value  

person = Person("Alice", 30)
print(person.name)  # Alice
print(person.age)   # 30
person.age = 31
print(person.age)   # 31
# person.__age = 32  # Raises AttributeError: 'Person' object has no attribute '__age'
# print(person.__age)  # Raises AttributeError: 'Person' object has no attribute '__age'
# person._name = "Bob"  # Allowed, but not recommended
print(person._name)  # Bob

# 5. Class methods and static methods
class MathUtils:
    @classmethod
    def add(cls, a, b):
        return a + b

    @staticmethod
    def multiply(a, b):
        return a * b 
print(MathUtils.add(5, 3))  # 8
print(MathUtils.multiply(5, 3))  # 15

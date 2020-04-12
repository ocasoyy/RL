# FM
import pandas as pd
import numpy as np
from collections import Counter

class Dog:
    def __init__(self, name):
        self.name = name
        self.tricks = []

    def add_trick(self, trick):
        self.tricks.append(trick)

    def __repr__(self):
        return "Dog named as {}".format(self.name)

    def __getitem__(self, position):
        return self.tricks[position]

dog = Dog("fido")
dog.tricks
dog.add_trick("trick1")
dog.tricks
dog[0]

class Reverse:
    def __init__(self, data):
        self.data = data
        self.index = len(data)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == 0:
            raise StopIteration
        self.index = self.index - 1
        return self.data[self.index]

rev = Reverse('spam')

d = next(rev)
d






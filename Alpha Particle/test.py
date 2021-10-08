

import numpy as np


def function(x):
    return x ** 3 + 2 * x ** 2 + 3 * x + 5


x = []
y = []
err = []
array = []
N = 1000

for counter in range(0, N):
    x.append(counter)
    y.append(function(counter))
    err.append(1)

for i in range(0,3):
    for counter in range(0,N):
        if i == 0:
            array.append(str(x[counter]))
        elif i == 1:
            array.append(str(y[counter]))
        elif i == 2:
            array.append(str(err[counter]))
        else:
            print("error")

file = open("data.txt","w")

for counter in range(0,len(array)):
    file.write(array[counter]+"/n")
file.close()
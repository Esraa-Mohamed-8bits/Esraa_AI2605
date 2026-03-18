from core.ndarray import Array
import math

def mean(arr):
    data = arr.flatten().data
    total = 0
    for x in data:
        total += x
    return total / len(data)

def var(arr):
    data = arr.flatten().data
    m = mean(arr)
    total = 0
    for x in data:
        total += (x - m) ** 2
    return total / len(data)

def std(arr):
    v = var(arr)
    return math.sqrt(v)

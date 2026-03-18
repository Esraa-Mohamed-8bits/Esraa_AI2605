from creation.array_creation import array, zeros, ones, eye
from statistic.stats import mean, var, std
from exceptions.errors import DimensionMismatchError, NonNumericDataError, InvalidShapeError, InvalidOperationError
from core.ndarray import Array

def main():

    print("zeros")
    z = zeros((2,2,2))
    print(z)


    print("\nones")
    o = ones((3,2))
    print(o)


    print("\neye")
    e = eye(3)
    print(e)


    print("\nArray creation")
    a = array([[1,2,3],[4,5,6]])
    print(a)


    print("\nflatten")
    f = a.flatten()
    print(f)


    print("\nStatistics")
    print("Mean:", mean(a))
    print("Variance:", var(a))
    print("Std:", std(a))


    print("\nAdd array")
    b = array([[10,20,30],[40,50,60]])
    print(a + b)


    print("\nAdd scalar")
    print(a + 5)


    print("\nMultiply scalar")
    print(a * 3)


    print("\npower")
    print(a ** 2)


    print("\nDimension mismatch")
    try:
        x = array([[1,2],[3,4]])
        y = array([[1,2,3],[4,5,6]])
        print(x + y)
    except DimensionMismatchError as e:
        print("Caught:", e)


    print("\nNon numeric data")
    try:
        ar_no = array([[1,2],["a",4]])
    except NonNumericDataError as e:
        print("Caught:", e)


    print("\nInvalid operation")
    try:
        print(a + "hello")
    except InvalidOperationError as e:
        print("Caught:", e)


    print("\nInvalid shape")
    try:
        zeros((2,-3))
    except Exception as e:
        print("Caught:", e)

if __name__ == "__main__":
    main()

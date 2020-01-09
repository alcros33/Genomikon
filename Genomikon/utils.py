""" Helper functions"""
from .core import *

def argmax(L: Iterable, key: Callable=None):
    if key:
        def _key(i): return key(L[i])
    else:
        def _key(i): return L[i]
    return max(range(len(L)), key=_key)

def argmin(L: Iterable, key: Callable=None):
    if key:
        def _key(i): return key(L[i])
    else:
        def _key(i): return L[i]
    return min(range(len(L)), key=_key)

def bits_to_array(bits: str, n: int, minv: float, maxv: float):
    """ Transofrms a bit string into a np.array with size n
        Maps each entry to range(minv, maxv)
    """
    arr = np.zeros(n)
    k = len(bits) // n
    for i in range(n):
        num = int(bits[i * k:(i + 1) * k], 2)
        num /= (2**k-1)
        arr[i] = num*(maxv-minv) + minv
    return arr

def bit_encoding_size(n: int, minv: float, maxv: float, prec: int = 6):
    """ Returns apropiate number of bits to encode n numbers in range(minv,maxv)
        with given precision
    """
    bits_per_number = int(np.ceil(np.log2((maxv - minv) * 10**prec)))
    return  bits_per_number*n

def random_range_bounds(low: int, high:int):
    """ Returns two numbers: left and right which describe a range given bounds
    """
    left, right = random.randint(low, high-1), random.randint(low+1, high)
    if left > right: left, right = right, left
    return left, right

def traveling_salesman_objective(val: Permutation, data):
    """ Returns the sum of the weights for a given permutation of the TSP
    Assumes it starts and ends in the node 0"""
    assert len(val) == len(data)
    res = 0
    for it, x in enumerate(val[:-1]):
        res += data[x][val[it+1]]
    return res + data[val[-1]][val[0]]

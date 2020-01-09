"""
Implements validators for genomes
Validators are functions that receive a value and return the "cleaned" value
There are two types of validators:
1.- Fatal: those who may raise GenValidationError
2.- Non-Fatal: those who don't raise exceptions
"""

from .core import *

__all__ = ["GenValidationError", "is_permutation", "bounds_validator"]

class GenValidationError(Exception):
    pass

def is_permutation(val: Permutation):
    """Checks if val is a valid permutation,
        Fatal
    """
    ok = (len(val) == len(set(val)))
    ok &= (max(val) == (len(val)-1))
    ok &= (min(val) == 0)
    if not ok:
        raise GenValidationError(f" {val} Not a valid permutation")
    return val

def bounds_validator(val: np.ndarray, bounds: Union[Size, Sizes]):
    """If bounds is a Size: checks that every element of array is within bounds,
        If bounds is list of Sizes: checks that every element is on its corresponding bound
        Non-Fatal
        Clips the elements to the max or min
    """
    if isinstance(bounds[0], (int, float)):
            return np.maximum(bounds[0], np.minimum(bounds[1], val))
    assert len(bounds) == len(val)
    return np.array([np.maximum(b[0], np.minimum(b[1], val[it])) for it, b in enumerate(bounds)])
    

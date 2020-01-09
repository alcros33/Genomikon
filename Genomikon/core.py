import sys, os, shutil, gc, subprocess, inspect, time
import csv, gzip, json, io, pickle
from collections import Counter, defaultdict, namedtuple, OrderedDict
from collections.abc import Iterable
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum, IntEnum
from functools import partial, reduce
from numbers import Number
from pathlib import Path
from typing import Any, Callable, Collection, Dict, Iterator, List, Mapping, NewType, Tuple, Union
import weakref
# Lib imports
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as random

PathOrStr = Union[Path, str]
Size = Tuple[float, float]
Sizes = List[Size]
Permutation = List[int]
np.set_printoptions(precision=5, suppress=True, threshold=50, edgeitems=4, linewidth=120)

def identity(x): return x

def copy_class(cls):
    return type(cls.__name__, cls.__bases__, dict(cls.__dict__))

def delegate_args(from_f: Callable, to_f: Callable):
    params_from = inspect.signature(from_f).parameters.values()
    sig_to = inspect.signature(to_f)
    to_f.__signature__ = sig_to.replace(parameters=params_from)

def functor(cls):
    """
    Decorator, transform a class into a functor
    Overrides the __new__ method and makes it call __func__ method
    __func__ method must be defined
    """
    if not inspect.ismethod(cls.__func__):
        cls.__func__ = classmethod(cls.__func__)
    cls.__new__ = lambda cls, *args, **kwargs: cls.__func__(*args, **kwargs)
    delegate_args(cls.__func__.__func__, cls.__new__)
    cls.__is_functor__ = True
    return cls

def delegates(to=None, keep=False):
    "Decorator: replace `**kwargs` in signature with params from `to`"
    def _f(f):
        if to is None:
            to_f, from_f = f.__base__.__init__, f.__init__
        else:
            to_f, from_f = to, f
        sig = inspect.signature(from_f)
        sigd = dict(sig.parameters)
        k = sigd.pop('kwargs')
        s2 = {k: v for k, v in inspect.signature(to_f).parameters.items()
              if v.default != inspect.Parameter.empty and k not in sigd}
        sigd.update(s2)
        if keep:
            sigd['kwargs'] = k
        from_f.__signature__ = sig.replace(parameters=sigd.values())
        return f
    return _f

CTX = dict()

@contextmanager
def set_context(**kwargs):
    """ Set a dict CTX containing information that can be used by functions called inside enclosure"""
    global CTX
    CTX = dict(**kwargs)
    try:
        yield
    finally:
        pass

def chunks(l: Collection, n: int, reflect: bool = False)->Iterable:
    "Yield successive `n`-sized chunks from `l`."
    for i in range(0, len(l), n):
        if i+n > len(l) and reflect:
            yield l[i:i + n] + l[:(i+n)-len(l)]
        else:
            yield l[i:i + n]

def ifnone(x, default):
    if x is None:
        return default
    return x

## Code below this point was copied from fast.ai github
## See https://github.com/fastai/fastai/blob/master/fastai/core.py for source
## See https://github.com/fastai/fastai/blob/master/LICENSE for license

def func_args(func):
    "Return the arguments of `func`."
    code = func.__code__
    return code.co_varnames[:code.co_argcount]

def num_cpus()->int:
    "Get number of cpus"
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()

def is_listy(x: Any)->bool: return isinstance(x, (tuple, list))
def is_tuple(x: Any)->bool: return isinstance(x, tuple)
def is_dict(x: Any)->bool: return isinstance(x, dict)
def is_pathlike(x: Any)->bool: return isinstance(x, (str, Path))

# def parallel(func, arr: Collection, max_workers: int = None, leave=False):
#     "Call `func` on every element of `arr` in parallel using `max_workers`."
#     max_workers = ifnone(max_workers, defaults.cpus)
#     if max_workers < 2:
#         results = [func(o, i) for i, o in progress_bar(
#             enumerate(arr), total=len(arr), leave=leave)]
#     else:
#         with ProcessPoolExecutor(max_workers=max_workers) as ex:
#             futures = [ex.submit(func, o, i) for i, o in enumerate(arr)]
#             results = []
#             for f in progress_bar(concurrent.futures.as_completed(futures), total=len(arr), leave=leave):
#                 results.append(f.result())
#     if any([o is not None for o in results]):
#         return results

class PrettyString(str):
    "Little hack to get strings to show properly in Jupyter."
    def __repr__(self): return self

@contextmanager
def working_directory(path: PathOrStr):
    "Change working directory to `path` and return to previous on exit."
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)

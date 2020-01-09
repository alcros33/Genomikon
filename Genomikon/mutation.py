"""
Mutators are defined here.
A Mutator is a callable that accepts a genotype
and returns the same genotype with (probably) mutations
"""
from .core import *
import Genomikon.core as core
from .genome import *

__all__ = ["Mutator", "BinaryUniformMutator", "PermutationSwapMutator", "PermutationInsertMutator",
            "PermutationDisplacementMutator", "FloatNonUniformMutator", "FloatBoundsMutator",
            "FloatUniformMutator", "ParameterBasedMutator"]

class Mutator:
    """ Base class for all types of mutators
    @param prob The mutation probability
    """
    genomeType = None

    def __init__(self, prob: float):
        self._prob = prob

    def __call__(self, value):
        return self.mutate(value)

def flip_bit(c): return "0" if c == "1" else "1"

# Binary mutations
class BinaryUniformMutator(Mutator):
    """Flips bits randomly given probability"""
    genomeType = BinaryGenome
    def mutate(self, value: str):
        res = ""
        for c in value:
            if random.random() < self._prob:
                res += flip_bit(c)
            else:
                res += c
        return res

# Permutation crossovers
class PermutationSwapMutator(Mutator):
    """Randomly swaps a value with other"""
    genomeType = PermutationGenome
    def mutate(self, value: Permutation):
        if  self._prob >= random.random():
            n1, n2 = random.randint(0, len(value)), random.randint(0, len(value))
            value[n1], value[n2] = value[n2], value[n1]
        return value

class PermutationInsertMutator(Mutator):
    """Randomly inserts a value in another position"""
    genomeType = PermutationGenome
    def mutate(self, value: Permutation):
        if self._prob >= random.random() :
            n1, n2 = random.randint(0, len(value)), random.randint(0, len(value))
            val = value.pop(n1)
            value.insert(n2, val)
        return value

class PermutationDisplacementMutator(Mutator):
    """ Performs Insert Mutation num_pos times"""
    genomeType = PermutationGenome
    def __init__(self, prob: float, num_pos:int=None):
        self._prob = prob
        self._num_pos = num_pos
    def mutate(self, value: Permutation):
        n = len(value)
        if random.random() > self._prob:
            return value
        if self._num_pos is None:
            num_pos = random.randint(0, n)
        else:
            num_pos = self._num_pos
        poped = [0 for _ in range(n)]
        positions = list(random.permutation(n))[:num_pos]
        for it, pos in enumerate(sorted(positions, reverse=True)):
            ip = random.randint(0, n)
            poped[it] = (ip, value.pop(pos))
        for ip, val in sorted(poped, reverse=True):
            value.insert(ip, val)
        return value

# Float crossovers
class FloatNonUniformMutator(Mutator):
    genomeType = FloatGenome
    def __init__(self, prob: float, low: float, high: float):
        self._prob = prob
        self._bounds = [low, high]
    
    def mutate(self, value: np.ndarray):
        if random.random() > self._prob:
            return value
        k = random.randint(0,len(value))
        r = random.random()
        ratio = float(core.CTX["GENERATION"]) / float(core.CTX["MAX_GENERATIONS"])
        if random.random() > 0.5:
            value[k] -= (value[k]-self._bounds[0])*(1-r**((1-ratio)*5))
        else:
            value[k] += (self._bounds[1]-value[k])*(1-r**((1-ratio)*5))
        return value

class FloatBoundsMutator(Mutator):
    genomeType = FloatGenome
    def __init__(self, prob: float, low: float, high: float):
        self._prob = prob
        self._bounds = [low, high]

    def mutate(self, value: np.ndarray):
        if random.random() > self._prob:
            return value
        k = random.randint(0,len(value))
        if random.random() > 0.5:
            value[k] = self._bounds[1]
        else:
            value[k] = self._bounds[0]
        return value

class FloatUniformMutator(Mutator):
    genomeType = FloatGenome
    def __init__(self, prob: float, low: float, high: float):
        self._prob = prob
        self._bounds = [low, high]
    def mutate(self, value: np.ndarray):
        if random.random() > self._prob:
            return value
        k = random.randint(0, len(value))
        value[k] = self._bounds[0] + random.random()*(self._bounds[1]-self._bounds[0])
        return value

class ParameterBasedMutator(Mutator):
    genomeType = FloatGenome
    def __init__(self, prob: float, low: float, high: float):
        self._prob = prob
        self._bounds = [low, high]
    
    def mutate(self, value:np.ndarray):
        if random.random() > self._prob:
            return value
        u = random.random()
        k = random.randint(0, len(value))
        d = min(value[k]-self._bounds[0], self._bounds[1]-value[k]) / (self._bounds[1]-self._bounds[0])
        eta = 100 + core.CTX["GENERATION"]
        if u > 0.5:
            dq = 1 - (2*(1-u)+2*(u-0.5)*(1-d)**(eta+1))**(1.0/(eta+1))
        else:
            dq = (2*u + (1-2*u)*(1-d)**(eta+1))**(1.0/(eta+1)) - 1
        value[k] += dq*(self._bounds[1]-self._bounds[0])
        return value
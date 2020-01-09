"""
Survivor Selectors are defined here
A survivor selector is a callable that given a list of parents and children,
returns two lists: the indexes of the parents and the indexes of the children
that will form the next generation
"""
from .core import *
from .genome import *

__all__ = ["SurvivorSelector", "MergeGenerationSelector", "ReplaceGenerationSelector", "UniformStateSelector"]

class SurvivorSelector:
    """ Base class for all  Survivor Selectors
    @param size The amount of survivors to be selected each generation
    """
    def __init__(self, size: int):
        self._size = size
    def __call__(self, parents: List[AbstractGenome], children: List[AbstractGenome]):
        return self.select(parents, children)

class MergeGenerationSelector(SurvivorSelector):
    """Merges the two populations and keeps the best"""
    def select(self, parents: List[AbstractGenome], children: List[AbstractGenome]):
        population = [(x, i) for i, x in enumerate(parents)]
        population += [(x, i+len(parents)) for i, x in enumerate(children)]
        population.sort(key=lambda x: x[0].fitness, reverse=True)
        pidx = [i for (x,i) in population[:self._size] if i < len(parents)]
        chidx = [i-len(parents) for (x,i) in population[:self._size] if i >= len(parents)]
        return pidx, chidx

class ReplaceGenerationSelector(SurvivorSelector):
    """Keeps the children only"""
    def select(self, parents: List[AbstractGenome], children: List[AbstractGenome]):
        return [], list(range(len(children)))

class UniformStateSelector(SurvivorSelector):
    """Replaces the worst k parents with the best k children"""
    def __init__(self, k: int):
        self._k = k
    def select(self, parents: List[AbstractGenome], children: List[AbstractGenome]):
        pidx = sorted([(x, i) for i, x in enumerate(parents)], key=lambda x: x[0].fitness, reverse=True)
        chidx = sorted([(x, i) for i, x in enumerate(children)], key=lambda x: x[0].fitness, reverse=True)
        return [i for (x, i) in pidx[:-self._k]], [i for (x, i) in chidx[:self._k]]
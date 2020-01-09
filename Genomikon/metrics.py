"""Metrics are defined here
A metric is a function that takes the population as an argument and returns a number
Metrics will be called and stored every generation
"""
from .core import *
from .genome import AbstractGenome

__all__ = ["max_fitness", "min_fitness", "mean_fitness", "std_fitness"]

def max_fitness(population: List[AbstractGenome]) -> float:
    return max(x.fitness for x in population)

def min_fitness(population: List[AbstractGenome]) -> float:
    return min(x.fitness for x in population)

def mean_fitness(population: List[AbstractGenome]) -> float:
    return np.mean([x.fitness for x in population])

def std_fitness(population: List[AbstractGenome]) -> float:
    return np.std([x.fitness for x in population])
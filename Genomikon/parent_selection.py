"""
Selectors are defined here
A selector is a callable that given a list of genomes,
returns the indexes of the parents (duplicate indexes are fine)
"""
from .core import *
from .genome import *

__all__ = ["Selector", "ProportionalSelector", "UniversalStochasticSelector",
            "DeterministicSamplingSelector", "TournamentSelector"]

class Selector:
    """ Base class for all Selectors
    @param size The amount of parents to be selected each generation
    """
    def __init__(self, size: int):
        self._size = size
    def __call__(self, population: List[AbstractGenome]):
        return self.select(population)

## Proportional Selectors

class ProportionalSelector(Selector):
    """ Selects using a probabilty distribution
    given the normalized values of each genome's fitness
    """
    def select(self, population):
        n = len(population)
        sum_fitness = sum(x.fitness for x in population)
        probabilty = [(x.fitness/sum_fitness) for x in population]
        return random.choice(range(n), size=self._size, replace=True, p=probabilty)

def get_expected_values(population: List, size: int, sigma_scale: bool):
    n = len(population)
    sum_fitness = sum(x.fitness for x in population)
    if not sigma_scale:
        return [(x.fitness/sum_fitness)*size for x in population]
    mean_fitness = sum_fitness/n
    sig = (1.0 / (n - 1)) * sum((x.fitness - mean_fitness) ** 2 for x in population)
    if sig == 0:
        return [1 for _ in range(n)]
    return [1 + (x.fitness - mean_fitness)/(2*sig) for x in population]

class UniversalStochasticSelector(Selector):
    def __init__(self, size: int, sigma_scale: bool = True):
        self._size = size
        self._sigma = sigma_scale
    def select(self, population):
        n = len(population)
        expected = get_expected_values(population, self._size, self._sigma)
        idx = []
        sum_ = 0
        ptr = random.random()
        for i in range(n):
            sum_ += expected[i]
            while sum_ > ptr:
                idx.append(i)
                ptr += 1
        return idx

class DeterministicSamplingSelector(Selector):
    """Selects parents deterministically using their expected values"""
    def __init__(self, size: int, sigma_scale: bool = True):
        self._size = size
        self._sigma = sigma_scale
    def select(self, population):
        expected = get_expected_values(population, self._size, self._sigma)
        idxs = []
        for it, p in enumerate(expected):
            for _ in range(int(p)):
                idxs.append(it)
        expected = sorted([p - int(p) for p in expected], reverse=True)
        idxs += expected[:self._size - len(idxs)]
        return idxs

## Tournament
class TournamentSelector(Selector):
    def __init__(self, size: int, tournament_size: int, prob: float = 1.0):
        self._size = size
        self._tournamentSize = tournament_size
        self._prob = prob

    def select(self, population):
        idx = []
        n = len(population)
        pop = [(x, i) for i, x in enumerate(population)]
        while len(idx) < self._size:
            random.shuffle(pop)
            for t in chunks(pop, self._tournamentSize):
                if len(t) < self._tournamentSize:
                    break
                if self._prob >= random.random():
                    _, best = max(t, key=lambda x: x[0].fitness)
                else:
                    _, best = min(t, key=lambda x: x[0].fitness)
                idx.append(best)
        return idx[:self._size]



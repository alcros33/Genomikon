""" Algorithm class which executes the genetic algorithm"""
from .callbacks import BaseCallback
from .core import *
from .genome import AbstractGenome
from .parent_selection import Selector
from .survivor_selection import SurvivorSelector
from .metrics import max_fitness

# Workaround
import Genomikon.core as core

__all__ = ['Algorithm']

class Algorithm:
    def __init__(self, population: List[AbstractGenome], parent_selector: Selector,
                survivor_selector: SurvivorSelector, metrics:Collection[Callable]=[],
                callbacks:Collection[BaseCallback]=[]):
        assert len(population) > 0
        """Class that runs the algoritm with given population"""
        self.__genome_type__ = population[0].__class__

        self.n = len(population)
        self.population = [x for x in population]
        for i in range(len(population)): self.population[i].evaluate()
        self.initialPopulation = [x.copy() for x in population]
        self.bests = []

        self.parentSelector = parent_selector
        self.survivorSelector = survivor_selector
        self.numParents = population[0]._crossFunc.num_parents
        self.numChildren = population[0]._crossFunc.num_children
        self.metrics = [max_fitness] + list(metrics)
        self.callbacks = sorted([C(self) for C in callbacks], key= lambda x: x.order)

    def fit(self, max_generations: int):
        """Alias for .run()"""
        return self.run(max_generations)
    
    def simulate(self, max_generations: int, iterations: int=1):
        """Runs for max_generations, ´iterations´ times and returns the best result"""
        for _ in range(iterations):
            self.run(max_generations)
        return max(self.bests, key=lambda x: x.fitness)

    def run(self, max_generations: int):
        self.population = [x.copy() for x in self.initialPopulation]
        self.metrics_record = []
        # Callbacks
        for C in self.callbacks: C.on_run_begin()
        for gen in range(max_generations):
            with core.set_context(MAX_GENERATIONS=max_generations, GENERATION=gen):
                gen_timer = time.perf_counter()
                # Callbacks
                for C in self.callbacks: C.on_generation_begin()

                ## Select indexes of parents
                # Callbacks
                for C in self.callbacks: C.on_selection_begin()
                parents_idxs = self.parentSelector(self.population)
                random.shuffle(parents_idxs)

                ## Generate child by crossing
                # Callbacks
                for C in self.callbacks: C.on_crossover_begin()
                children = []
                for p in chunks(parents_idxs, self.numParents, True):
                    children += self.population[p[0]].cross(*[self.population[idx] for idx in p[1:]])

                ## Mutate and validate children
                # Callbacks
                for C in self.callbacks: C.on_mutation_begin()
                for i in range(len(children)):
                    children[i].mutate().validate()
                
                ## Evaluate
                # Callbacks
                for C in self.callbacks: C.on_evaluation_begin()
                for i in range(len(children)):
                    children[i].evaluate()                

                ## Select survivors
                # Callbacks
                for C in self.callbacks: C.on_survivor_begin()
                pidx, chidx = self.survivorSelector(self.population, children)
                self.population = [self.population[i] for i in pidx] + [children[i] for i in chidx]

                self.bests.append(max(self.population, key=lambda x: x.fitness))
                
                ## Metrics
                self.metrics_record.append({m.__name__:m(self.population) for m in self.metrics})
                self.metrics_record[-1]["time"] = time.perf_counter() - gen_timer
                

                # Callbacks
                for C in self.callbacks: C.on_generation_end()
        # Callbacks
        for C in self.callbacks: C.on_run_end()

        return max(self.bests, key=lambda x: x.fitness)

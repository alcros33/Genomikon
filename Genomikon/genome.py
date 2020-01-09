"""
Implements a collection of genome types
"""
from .core import *
from .validators import is_permutation, GenValidationError

__all__ = ["genome_operator", "AbstractGenome", "GenomeType", "BinaryGenome",
           "FloatGenome", "PermutationGenome"]

class GenomeGenerator:
    """ Proxy class that assigns operators to the genome type """
    def __init__(self, genome_type, *args, **kwargs):
        self.genomeType = copy_class(genome_type)
        self.genomeName = self.genomeType.__name__
        self._args, self._kwargs = args, kwargs

    def __getattr__(self, name):
        attr = getattr(self.genomeType, name)
        if not (callable(attr) and hasattr(attr, "__is_gop__")):
            raise AttributeError

        def _proxy_op_setter(op):
            if hasattr(op, "genome_type"):
                if getattr(op, "genome_type").__name__ != self.genomeName:
                    raise GenValidationError(
                        f"Gen operator cannot be used on {self.genomeName}")
            setattr(self.genomeType, f"_{name}Func", op)
            return self
        return _proxy_op_setter

    def population(self, n: int):
        assert n > 0
        population = [self.genomeType.random(*self._args, **self._kwargs) for _ in range(n)]
        for i in range(n): population[i].evaluate()
        return population

def genome_operator(_f):
    """Decorates operators of a genome"""
    _f.__is_gop__ = True
    return _f

class AbstractGenome():
    """Base class for all genomes, do not use directly"""
    def __str__(self):
        return repr(self)
    
    def __repr__(self):
        res = f'<{self.__class__.__name__}: value={str(self.value)}'
        if hasattr(self, "fitness"):
            res += f' fitness={self.fitness:.4f}'
        return res +'>'

    def __eq__(self, other): return self.value == other.value
    def __ne__(self, other): return not (self == other)
    def __lt__(self, other): return self.value < other.value
    def __ge__(self, other): return not(self < other)
    def __gt__(self, other): return self.value > other.value
    def __le__(self, other): return not(self > other)

    def copy(self):
        cls = self.__class__
        new_args = {k: deepcopy(getattr(self, k)) for k in self._copyNew}
        New = cls(**new_args)
        if hasattr(self, "fitness"): New.fitness = deepcopy(self.fitness)
        return New

    def __add__(self, others):
        """ Shorcut for cross"""
        raise self.cross(*others)

    # Operators
    @genome_operator
    def cross(self, *others):
        """Cross and produce offspring"""
        return self.__class__._crossFunc(self, *others)

    @genome_operator
    def mutate(self):
        """ Perform Mutation"""
        self.value = self.__class__._mutateFunc(self.value)
        return self

    @genome_operator
    def validate(self):
        """Perform validation"""
        if hasattr(self, "_validateFunc"):
            self.value = self.__class__._validateFunc(self.value)
        return self

    @genome_operator
    def evaluate(self):
        """Evaluate on the objective function"""
        self.fitness = self.__class__._evaluateFunc(self.value)
        return self.fitness

    @classmethod
    def generator(cls, *args, **kwargs):
        """Creates a GenomeGenerator object
           Generator will pass down args and kwargs to .random method
        """
        return GenomeGenerator(cls, *args, ** kwargs)

def GenomeType(cls):
    """ Decorator that transforms a class into a GenomeType
        All genome types MUST be decorated with it
    """
    if not issubclass(cls, AbstractGenome):
        cls = type(cls.__name__, (AbstractGenome,) +
                   cls.__bases__, dict(cls.__dict__))
    delegate_args(cls.random.__func__, cls.generator.__func__)
    cls = dataclass(cls, repr=False, eq=False)
    cls._copyNew = list(cls.__dataclass_fields__.keys())
    return cls

@GenomeType
class BinaryGenome:
    """ Genome represented by a binary string"""
    value: str

    @classmethod
    def random(cls, size: int):
        return cls("".join(np.random.randint(2,size=(size,)).astype(str)))

@GenomeType
class FloatGenome:
    """ Genome represented by a np array of float32 """
    value: np.ndarray

    def __post_init__(self):
        self.value = self.value.astype(np.float_)

    @classmethod
    def random(cls, size: int, bounds: Union[Size, Sizes]):
        if isinstance(bounds[0], (int, float)):
            return cls(random.uniform(*bounds, size=size))
        return np.array([random.uniform(*b) for b in bounds])

@GenomeType
class PermutationGenome:
    """ A genome that contains a permutation"""
    value: Permutation

    def __post_init__(self):
        self.value = is_permutation(self.value)

    @classmethod
    def random(cls, size: int):
        return cls(list(random.permutation(size)))

"""
Crossover operators are defined here.
A Cross is a callable that accepts a number of parents (defined in his num_parents var)
and return a list of its generated offspring (defined in his num_children var)
"""
from .core import *
from .genome import *
from .utils import random_range_bounds

__all__ = ['Cross', 'NoCross', 'BinaryUniformCross', 'BinaryOnePointCross','BinaryTwoPointCross',
        'FloatOnePointCross', 'FloatUniformCross', 'FloatMiddleCross', 'FloatSimulatedBinaryCross', 'FloatRecombinationCross', 'FloatHeuristicCross', 'PermutationOrderCross', 'PermutationPartiallyMappedCross', 'PermutationPositionBasedCross', 'PermutationOrderBasedCross']

class Cross:
    """ Base class for all types of cross
    @param prob The cross probability
    """
    genome_type = None
    num_parents = 2
    num_children = 2
    def __init__(self, prob: float):
        self._prob = prob
    def __call__(self, *args) -> List[AbstractGenome]:
        if random.random() > self._prob:
            return [x.copy() for x in args]
        return self.cross(*args)

class NoCross(Cross):
    def cross(self, A, B):
        return [A,B]

# Binary crossovers
class BinaryUniformCross(Cross):
    genome_type = BinaryGenome

    def cross(self, A, B):
        cls = A.__class__
        S = random.randint(0,2, len(A.value)).astype(bool)
        C = cls("".join(A.value[it] if val else B.value[it] for it,val in enumerate(S)))
        D = cls("".join(B.value[it] if val else A.value[it] for it,val in enumerate(S)))
        return [C,D]

class BinaryOnePointCross(Cross):
    genome_type = BinaryGenome

    def cross(self, A, B):
        cls = A.__class__
        pos = random.randint(0, len(A.value))
        C = cls(A.value[:pos] + B.value[pos:])
        D = cls(B.value[:pos] + A.value[pos:])
        return [C, D]

class BinaryTwoPointCross(Cross):
    genome_type = BinaryGenome

    def cross(self, A, B):
        cls = A.__class__
        n = len(A.value)
        left, right = random_range_bounds(0, len(A.value))
        C = cls(A.value[:left] + B.value[left:right] + A.value[right:])
        D = cls(B.value[:left] + A.value[left:right] + B.value[right:])
        return [C, D]

# Float crossover
class FloatOnePointCross(Cross):
    genome_type = FloatGenome
    def cross(self, A, B):
        cls = A.__class__
        pos = random.randint(0, len(A.value))
        C = cls(np.concatenate(A.value[:pos], B.value[pos:]))
        D = cls(np.concatenate(B.value[:pos], A.value[pos:]))
        return [C, D]

class FloatUniformCross(Cross):
    genome_type = FloatGenome
    def cross(self, A, B):
        cls = A.__class__
        S = random.randint(0,2, len(A.value)).astype(bool)
        C = cls(np.array([A.value[it] if val else B.value[it] for it,val in enumerate(S)]))
        D = cls(np.array([B.value[it] if val else A.value[it] for it,val in enumerate(S)]))
        return [C,D]

class FloatMiddleCross(Cross):
    genome_type = FloatGenome

    def __init__(self, prob: float, alpha: float=0.5):
        self._prob = prob
        self._alpha = alpha

    def cross(self, A, B):
        cls = A.__class__
        pos = random.randint(0, len(A.value))
        othA = A[:pos]*(1-self._alpha) + B[:pos]*self._alpha
        othB = B[:pos]*(1-self._alpha) + A[:pos]*self._alpha
        C = cls(np.concatenate(A.value[:pos], othA))
        D = cls(np.concatenate(B.value[:pos], othB))
        return [C, D]

class FloatSimulatedBinaryCross(Cross):
    genome_type = FloatGenome

    def __init__(self, prob: float, eta: float=1):
        self._prob = prob
        self._eta = eta
    
    def cross(self, A, B):
        cls = A.__class__
        u = random.random()
        if u > 0.5: b = 1.0/(2*(1-u))
        else: b = 2*u
        b **= (1.0/(self._eta+1))
        P, M = A.value+B.value, B.value - A.value
        C = cls(0.5*(P - b*abs(M)))
        D = cls(0.5*(P + b*abs(M)))
        return [C,D]

class FloatRecombinationCross(Cross):
    genome_type = FloatGenome
    def __init__(self, prob: float, num_pos:int=None):
        self._prob = prob
        self._num_pos = num_pos
    
    def cross(self, A, B):
        cls = A.__class__
        n = len(A.value)
        if self._num_pos is None:
            num_pos = random.randint(0, n)
        else:
            num_pos = self._num_pos
        positions = list(random.permutation(n))[:num_pos]
        C, D = A.value.copy(), B.value.copy()
        C[positions] = (A[positions] + B[positions])/2.0
        D[positions] = (A[positions] + B[positions])/2.0
        return [cls(C), cls(D)]

class FloatHeuristicCross(Cross):
    genome_type = FloatGenome
    num_children = 1

    def cross(self, A, B):
        if A.fitness >= B.fitness:
            return [A.value + random.random()*(A.value-B.value)]
        return [B.value + random.random()*(B.value-A.value)]

class FloatAverageCross(Cross):
    genome_type = FloatGenome
    num_children = 1

    def cross(self, A, B):
        return [A.__class__((A.value+B.value)/2.0)]

# Permutation crossovers
class PermutationOrderCross(Cross):
    genome_type = PermutationGenome

    def cross(self, A, B):
        cls = A.__class__
        l1, r1 = random_range_bounds(0, len(A.value))
        S1 = A.value[l1:r1]
        P1 = [x for x in B.value if x not in S1]
        l2, r2 = random_range_bounds(0, len(A.value))
        S2 = B.value[l2:r2]
        P2 = [x for x in B.value if x not in S2]
        C = cls(P1[:l1] + S1 + P1[l1:])
        D = cls(P2[:l2] + S2 + P2[l2:])
        return [C, D]

class PermutationPartiallyMappedCross(Cross):
    genome_type = PermutationGenome

    def cross(self, A, B):
        cls = A.__class__
        left, right = random_range_bounds(0, len(A.value))
        S1, S2 = A.value[left:right], B.value[left:right]
        P1 = A.value[:left] + S2 + A.value[right:]
        P2 = B.value[:left] + S1 + B.value[right:]
        for x in filter(lambda x: x not in S1, S2):
            ia, ib = A.value.index(x), B.value.index(x)
            P1[ia] = A[ib]
        for x in filter(lambda x: x not in S2, S1):
            ia, ib = A.value.index(x), B.value.index(x)
            P2[ib] = B[ia]
        return [cls(P1), cls(P2)]

class PermutationPositionBasedCross(Cross):
    genome_type = PermutationGenome
    def __init__(self, prob: float, num_pos:int=None):
        self._prob = prob
        self._num_pos = num_pos

    def cross(self, A, B):
        cls = A.__class__
        n = len(A.value)
        if self._num_pos is None:
            num_pos = random.randint(0, n)
        else:
            num_pos = self._num_pos
        posA = list(random.permutation(n))[:num_pos]
        posB = list(random.permutation(n))[:num_pos]
        SA, SB = [A.value[i] for i in posA], [B.value[i] for i in posB]
        P1, P2 = [x for x in B.value if x not in SA], [x for x in A.value if x not in SB]

        for i in posA: P1.insert(i, A.value[i])
        for i in posB: P2.insert(i, B.value[i])
        return [cls(P1), cls(P2)]

class PermutationOrderBasedCross(Cross):
    genome_type = PermutationGenome
    def __init__(self, prob: float, num_pos:int=None):
        self._prob = prob
        self._num_pos = num_pos

    def cross(self, A, B):
        cls = A.__class__
        n = len(A.value)
        if self._num_pos is None:
            num_pos = random.randint(0, n)
        else:
            num_pos = self._num_pos
        posA = list(random.permutation(n))[:num_pos]
        posB = list(random.permutation(n))[:num_pos]
        P1, P2 = B.value, A.value
        for i, j in zip(posA, posB):
            P1[P1.index(A.value[i])] = -1
            P2[P2.index(B.value[i])] = -1
        ia, ib = 0,0
        for i in range(n):
            if P1[i] == -1:
                P1[i] = A.value[posA[ia]]
                ia += 1
            if P2[i] == -1:
                P2[i] = B.value[posB[ib]]
                ib += 1
        return [cls(P1), cls(P2)]

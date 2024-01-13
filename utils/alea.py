from random import random
from numpy import array as A
from numpy.random import choice
from typing import Any, Iterable


class Sampler:
    def __init__(self, samples: Iterable, distribution: Iterable = None):
        self._samples = samples
        self._distribution = A(distribution if distribution is not None else [1] * len(samples), dtype=float)
        self._distribution /= self._distribution.sum()

    def __call__(self):
        return choice(self._samples, p=self._distribution)


if __name__ == '__main__':
    s = Sampler(list(range(10)))


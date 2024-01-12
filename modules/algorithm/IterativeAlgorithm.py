import numpy as np
from abc import ABC, abstractmethod

class IterativeAlgorithm(ABC):
    def __init__(self, max_iter : int) -> None:
        self.max_iter = max_iter
        self.iterations = 0

    @abstractmethod
    def initialize(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def one_step(self) -> None:
        pass
    
    @abstractmethod
    def solve(self, *args, **kwargs) -> np.array:
        pass

    @abstractmethod
    def get_result(self) -> np.array:
        pass

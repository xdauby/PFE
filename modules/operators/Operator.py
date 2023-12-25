from abc import ABC, abstractmethod
import numpy as np

class Operator(ABC):

    @abstractmethod
    def transform(self, x : np.array) -> np.array:
        pass

    @abstractmethod
    def transposed_transform(self, y : np.array) -> np.array:
        pass

import numpy as np
import matplotlib.pyplot as plt
from modules.operators.Operator import Operator
import astra


class Projection(Operator):

    def __init__(self, proj_id) -> None:
        self.proj_id = proj_id

    def transform(self, x : np.array) -> np.array:
        Ax = astra.creators.create_sino(x, self.proj_id)[1]
        return Ax

    def transposed_transform(self, y : np.array) -> np.array:
        ATy = astra.creators.create_backprojection(y, self.proj_id)[1]
        return ATy

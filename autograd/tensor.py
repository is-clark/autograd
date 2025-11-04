from __future__ import annotations

import numpy as np
# import numpy.typing as npt
from function import *


class Tensor:
    def __init__(self, data:np.ndarray, requires_grad:bool=False, _ctx=None) -> None:
        self.data = data
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) # should I do this or leave it as None?
        self._ctx = _ctx

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)
    
    def __repr__(self):
        tensor_repr = np.array2string(self.data, separator=', ', prefix='Tensor(', suffix=')')
        return f"Tensor({tensor_repr}, requires_grad={self.requires_grad})"
    
    def __add__(self, other: Tensor) -> Tensor:
        return Add.apply(self, other)
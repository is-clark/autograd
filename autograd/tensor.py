from __future__ import annotations

import numpy as np
# import numpy.typing as npt
import autograd.function as F


class Tensor:
    def __init__(self, data:np.ndarray, requires_grad:bool=False, _ctx=None) -> None:
        self.data = data
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) # should I do this or leave it as None?
        self._ctx = _ctx

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

    def backward(self):
        if not self.requires_grad:
            raise RuntimeError("Can't call backwards() on a Tensor that does not require grad")

        topo_order = []
        visited = set()
        
        def build_topo(t: Tensor):
            if t not in visited:
                visited.add(t)
                if t._ctx:
                    for parent in t._ctx.parents:
                        build_topo(parent)
                    topo_order.append(t)

        build_topo(self)

        self.grad = np.ones_like(self.data)

        for t in reversed(topo_order):
            grad_stream = t.grad

            ctx = t._ctx

            parent_grads = ctx.backward(grad_stream)

            for parent, grad in zip(ctx.parents, parent_grads):
                if parent.requires_grad:
                    parent.grad += grad
    
    def __repr__(self):
        tensor_repr = np.array2string(self.data, separator=', ', prefix='Tensor(', suffix=')')
        return f"Tensor({tensor_repr}, requires_grad={self.requires_grad})"
    
    def __add__(self, other: Tensor) -> Tensor:
        return F.Add.apply(self, other)

    def __mul__(self, other: Tensor) -> Tensor:
        return F.Mul.apply(self, other)

    def __matmul__(self, other:Tensor) -> Tensor:
        return F.Matmul.apply(self.data, other)

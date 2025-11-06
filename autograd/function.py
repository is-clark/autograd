import numpy as np
from autograd.tensor import Tensor
from abc import ABC, abstractmethod

class Function(ABC):
    def __init__(self, parents:tuple[Tensor, ...] ) -> None:
        self.parents = parents
        
    @classmethod
    def apply(cls, *parents:Tensor):
        ctx = cls(parents)

        data = [t.data for t in parents]
        raw_out = ctx.forward(*data)

        requires_grad = any(t.requires_grad for t in parents)

        return Tensor(raw_out, requires_grad=requires_grad, _ctx=ctx)
    
    @abstractmethod
    def forward(self, *args:np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, grad_stream:np.ndarray) -> tuple[np.ndarray, ...]:
        pass

class Add(Function):
    def forward(self, x, y):
        return x + y

    def backward(self, grad_stream:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (grad_stream, grad_stream) 

class Mul(Function):
    def forward(self, x, y):
        self.x = x
        self.y = y
        return x * y

    def backward(self, grad_stream:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        grad_x = self.y * grad_stream
        grad_y = self.x * grad_stream
        return (grad_x, grad_y)

import numpy as np
from tensor import Tensor
from abc import abstractmethod

class Function:
    def __init__(self) -> None:
        self.parents:tuple[Tensor, ...]
        
    @classmethod
    def apply(cls, *tensors:Tensor):
        ctx = cls()

        ctx.parents = tensors

        data = [t.data for t in tensors]
        raw_out = ctx.forward(*data)

        requires_grad = any(t.requires_grad for t in tensors)

        return Tensor(raw_out, requires_grad=requires_grad, _ctx=ctx)
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def backward(self, *args, **kwargs):
        raise NotImplementedError

class Add(Function):
    def forward(self, x, y):
        return x + y
    
    def backward(self, grad_stream):
        # should I just do each individually since there are only two parents in the multiplication operation?
        for parent in self.parents:
            if parent.requires_grad:
                parent.grad += grad_stream
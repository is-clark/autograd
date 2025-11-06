import numpy as np
from autograd.tensor import Tensor
from autograd.function import Mul, Add # Make sure Mul is imported

# 1. Create leaf Tensors
a = Tensor(np.array([2.0, 3.0]), requires_grad=True)
b = Tensor(np.array([4.0, 5.0]), requires_grad=True)

print(f"a: {a}")
print(f"b: {b}")

# 2. Forward pass (c = a * b)
c = a * b
print(f"c: {c}")

# 3. Backward pass
# We pass an upstream gradient of 1s
c.backward(np.array([1.0, 1.0]))

# 4. Check gradients
# dL/da = dL/dc * dc/da = [1., 1.] * b = [1., 1.] * [4., 5.] = [4., 5.]
print(f"a.grad: {a.grad}") # Should be [4., 5.]

# dL/db = dL/dc * dc/db = [1., 1.] * a = [1., 1.] * [2., 3.] = [2., 3.]
print(f"b.grad: {b.grad}") # Should be [2., 3.]

import numpy as np
import numpy.testing as npt
from autograd.tensor import Tensor
from autograd.function import Mul, Add, Matmul 

def test_mul():
    print("--- Running Test: Mul ---")
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
    print(f"a.grad: {a.grad}") 
    npt.assert_almost_equal(a.grad, np.array([4., 5.]))

    # dL/db = dL/dc * dc/db = [1., 1.] * a = [1., 1.] * [2., 3.] = [2., 3.]
    print(f"b.grad: {b.grad}") 
    npt.assert_almost_equal(b.grad, np.array([2., 3.]))
    print("✓ Test Mul Passed")


def test_matmul():
    print("\n--- Running Test: Matmul ---")
    # 1. Create leaf Tensors
    # a: (2, 3)
    a_data = np.array([[1., 2., 3.], [4., 5., 6.]])
    a = Tensor(a_data, requires_grad=True)
    
    # b: (3, 2)
    b_data = np.array([[7., 8.], [9., 10.], [11., 12.]])
    b = Tensor(b_data, requires_grad=True)

    print(f"a (2,3): \n{a}")
    print(f"b (3,2): \n{b}")

    # 2. Forward pass (c = a @ b)
    # c: (2, 2)
    c = a @ b 
    print(f"c (2,2): \n{c}")
    
    # Expected forward pass result
    # c = [[58., 64.], [139., 154.]]
    c_expected = np.array([[58., 64.], [139., 154.]])
    npt.assert_almost_equal(c.data, c_expected)

    # 3. Backward pass
    # Upstream gradient grad_stream is (2, 2)
    grad_stream = np.array([[1., 1.], [1., 1.]])
    c.backward(grad_stream)

    # 4. Check gradients
    
    # a.grad = grad_stream @ b.T
    # (2, 2) @ (2, 3) = (2, 3)
    # [[1, 1], [1, 1]] @ [[7, 9, 11], [8, 10, 12]] = [[15, 19, 23], [15, 19, 23]]
    a_grad_expected = np.array([[15., 19., 23.], [15., 19., 23.]])
    print(f"a.grad: \n{a.grad}")
    npt.assert_almost_equal(a.grad, a_grad_expected)

    # b.grad = a.T @ grad_stream
    # (3, 2) @ (2, 2) = (3, 2)
    # [[1, 4], [2, 5], [3, 6]] @ [[1, 1], [1, 1]] = [[5, 5], [7, 7], [9, 9]]
    b_grad_expected = np.array([[5., 5.], [7., 7.], [9., 9.]])
    print(f"b.grad: \n{b.grad}")
    npt.assert_almost_equal(b.grad, b_grad_expected)
    print("✓ Test Matmul Passed")


# --- Run the tests ---
if __name__ == "__main__":
    test_mul()
    test_matmul()

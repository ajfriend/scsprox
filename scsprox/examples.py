import numpy as np
import cvxpy as cp


def print_file():
    import os
    print('\nRunning package from:', os.path.abspath(__file__))


def example():
    x = cp.Variable()
    obj = cp.Minimize(x)
    cons = [x >= 0]
    prob = cp.Problem(obj, cons)
    x_vars = dict(x=x)
    return prob, x_vars


def example2():
    x = cp.Variable(3)
    y = cp.Variable(2)
    prob = cp.Problem(cp.Minimize(cp.norm(x) + cp.norm(2 * y)))
    x_vars = dict(x=x, y=y)
    return prob, x_vars


def example3():
    x = cp.Variable(3)
    y = cp.Variable()
    prob = cp.Problem(cp.Minimize(cp.norm(x) + cp.norm(2 * y)))
    x_vars = dict(x=x, y=y)
    return prob, x_vars


def example_rand(m=10, n=5, seed=0):
    # Generate data.
    assert m > n
    np.random.seed(seed)
    A = np.random.randn(m, n)
    b = np.random.randn(m)

    x = cp.Variable(n)
    y = cp.Variable(m)
    z = cp.Variable()

    obj = cp.sum_squares(A @ x - b) + cp.norm(A.T @ y - x) + 0.1 * cp.norm(y) + cp.norm(z - y)
    prob = cp.Problem(cp.Minimize(obj))
    x_vars = dict(x=x, y=y, z=z)

    prob.solve(solver='ECOS')
    true_sol = dict(x=x.value, y=y.value, z=z.value)
    for k in true_sol:
        true_sol[k] = np.array(true_sol[k]).flatten()
        if len(true_sol[k]) == 1:
            true_sol[k] = true_sol[k][0]

    return prob, x_vars, true_sol

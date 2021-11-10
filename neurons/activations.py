import numpy as np


def f1(x, deriv=False):
    if not deriv:
        return np.where(x > 0, 0.5 * np.log(np.square(x) + 1), 0)
    return np.where(x > 0, x / (1 + np.square(x)), 0)
    
def f2(x, deriv=False):
    if not deriv:
        return np.where(x > 0, 2 / np.pi * np.arctan(x), 0)
    return np.where(x > 0, 2 / np.pi / (1 + np.square(x)), 0)
    
def f3(x, deriv=False):
    absx = 1 + np.abs(x)
    if not deriv:
        return np.maximum(x / absx, 0)
    return np.heaviside(x, 0) / absx
    
def f4(x, deriv=False):
    if not deriv:
        return np.maximum(np.tanh(x), 0)
    return np.heaviside(x, 0) / np.square(np.cosh(x))
    
def f5(x, deriv=False):
    squ = np.square(x)
    if not deriv:
        return np.where(x > 0, squ / (squ + 1), 0)
    return np.where(x > 0, 2 * x / np.square(1 + squ), 0)
    
def exp1(x, deriv=False):
    if not deriv:
        return np.where(x > 0, 1 - np.exp(-x), 0)
    return np.where(x > 0, np.exp(-x), 0)

def sigmoid(x, deriv=False):
    if not deriv:
        y = np.where(x >= 0, np.exp(-x), np.exp(x))
        return y / (1 + y)
    else:
        return sigmoid(x) * (1 - sigmoid(x))


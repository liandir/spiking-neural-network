import numpy as np


def f1(x, deriv=False):
    if not deriv:
        return 0.5 * np.log(np.square(x) + 1)
    else:
        return x / (1 + np.square(x))
    
def f2(x, deriv=False):
    if not deriv:
        return 2 / np.pi * np.arctan(x)
    else:
        return 2 / np.pi / (1 + np.square(x))
    
def f3(x, deriv=False):
    absx = 1 + np.abs(x)
    if not deriv:
        return x / absx
    else:
        return 1 / absx
    
def f4(x, deriv=False):
    if not deriv:
        return np.tanh(x)
    else:
        return 1 / np.square(np.cosh(x))

def f5(x, deriv=False):
    if not deriv:
        return np.tanh(x)
    else:
        return 1 / np.square(np.cosh(x))
    
def f6(x, deriv=False):
    if not deriv:
        squ = np.square(x)
        return np.where(x > 0, squ / (squ + 1), 0)
    else:
        return np.where(x > 0, 2 * x / np.square(1 + np.square(x)), 0)


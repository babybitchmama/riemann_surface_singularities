import numpy as np
import matplotlib
import doctest

"""Find a way to store pos and v, simulate it all at once (with animations)"""

def cmplx_div(z: tuple, w: tuple) -> tuple:
    '''The second part of each tuple is imaginary

    >>> cmplx_div((1, 0), (0, 1))
    (0.0, -1.0)
    '''
    a, b = z
    c, d = w

    v1 = (a*c + b*d) / (c**2 + d**2)
    v2 = (b*c - a*d) / (c**2 + d**2)
    return (v1, v2)

def cmplx_mult(z: tuple, w: tuple):
    '''The second part of each tuple is imaginary

    >>> cmplx_mult((0, 1), (0,1))
    (-1, 0)
    '''
    a, b = z
    c, d = w

    v1 = a*c - b*d
    v2 = a*d + b*c
    return (v1, v2)

def plus(z, w):
    z1, z2 = z
    w1, w2 = w
    new_1 = w1 + z1
    new_2 = z2 + w2

    return (new_1, new_2)

def main():
    dt = 0.001
    pos = (1, 0.1)
    v = (-1, 0)
    beta = 0.1
    a = 0

    for iter in range(1000):
        new_pos = plus(pos, cmplx_mult((dt, 0), (v)))
        a = cmplx_div(cmplx_mult((1-beta, 0), cmplx_mult(v, v)), pos)
        v = plus(v, cmplx_mult((dt, 0), a))
        pos = new_pos

        #print('a:', a)
       # print('v:', v)
    print('pos:', pos)
    print('v:', v)
    print('a:', a)
        

main()
print(doctest.testmod())

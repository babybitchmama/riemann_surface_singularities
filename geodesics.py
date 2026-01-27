import numpy as np
import matplotlib

def gamma_tt(beta, z, gamma_x, gamma_y, v1, v2, i):
    term_1 = ((1 - beta) / 2*z) 
    term_2 = gamma_x - np.i * gamma_y
    return term_1 * term_2

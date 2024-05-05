import numpy as np


def init_model_constpos(D, M, sig_x, sig_y):
    L = D - M
    model = {}
    model['A'] = np.identity(D)

    if (L > 0):
        model['C'] = np.zeros((M,D))
        model['C'][:M,:M] = np.identity(M)
    else:
        model['C'] = np.identity(M)

    model['Q'] = np.square(sig_x) * np.identity(D)
    model['R'] = np.square(sig_y) * np.identity(M)

    x0 = np.zeros(D)
    P0 = np.identity(D)
    return model, x0, P0
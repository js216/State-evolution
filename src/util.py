import cupy as cp
import numpy as np
# from functools import reduce
from scipy.linalg import expm

def reduce(fun, arr):
    arr_tmp = cp.empty(arr[0,:].shape)
    while len(arr) > 1:
        if len(arr) % 2 != 0:
            arr_tmp = arr[-1]
            arr = fun(arr[:-1][::2, ...], arr[:-1][1::2, ...])
            arr[-1] = arr[-1]@arr_tmp
        else:
            arr = fun(arr[::2, ...], arr[1::2, ...])
    return arr[0,...]

class PropOp:
    def __init__(self, H, batch_size):
        self.H = cp.array(np.repeat(
           -1j * 2*np.pi * H[np.newaxis,:,:,:], batch_size, axis=0))

        self.array_book_reduce = cp.empty(H.shape[1:-2])

    def init_dU(self):
       self.dU = cp.array(np.repeat(
          np.identity(self.H.shape[-2])[np.newaxis,:,:], self.H.shape[0], axis=0))

    def evolve(self, field, dt, s):
        dU = cp.einsum("ax,aijx->aij", field, self.H[:field.shape[0],:,:,1:], dtype=complex)+self.H[:field.shape[0],:,:,0]
        dU = cp.einsum("a, aij ->aij", dt, dU, dtype=complex)
        dU = expm_arr(dU, s)
        self.dU = reduce(cp.matmul, self.dU[::-1]) @ dU

    def get(self):
        return cp.asnumpy(reduce(cp.matmul, self.dU[::-1]))

def expm_arr(A_arr, s):
    # calculate powers of A
    A2 = A_arr @ A_arr
    A4 = A2 @ A2
    A6 = A4 @ A2
    A8 = A6 @ A2
    ident = cp.eye(A_arr.shape[-2], A_arr.shape[-1], dtype=complex)

    # calculate U, V
    b = (64764752532480000., 32382376266240000., 7771770303897600.,
            1187353796428800., 129060195264000., 10559470521600.,
            670442572800., 33522128640., 1323241920., 40840800., 960960.,
            16380., 182., 1.)

    B  = A_arr  * 2**-s
    B2 = A2 * 2**(-2*s)
    B4 = A4 * 2**(-4*s)
    B6 = A6 * 2**(-6*s)

    U2 = B6 @ (b[13]*B6 + b[11]*B4 + b[9]*B2)
    V2 = B6 @ (b[12]*B6 + b[10]*B4 + b[8]*B2)

    U  = B @ (U2 + b[7]*B6 + b[5]*B4 + b[3]*B2 + b[1]*ident[cp.newaxis,:,:])
    V  =      V2 + b[6]*B6 + b[4]*B4 + b[2]*B2 + b[0]*ident[cp.newaxis,:,:]

    # solve Px = Q
    X = cp.linalg.solve(V-U, V+U)

    # square s times
    for i in range(s):
        X = X @ X

    return X

def eval_num(expr, scope):
    return eval(expr, {**scope, 'np':np})

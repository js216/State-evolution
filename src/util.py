import numpy as np
import tensorflow as tf

b = (64764752532480000., 32382376266240000., 7771770303897600.,
      1187353796428800., 129060195264000., 10559470521600.,
      670442572800., 33522128640., 1323241920., 40840800., 960960.,
      16380., 182., 1.)

def expm_arr_tf(A_arr, s):
    # calculate powers of A
    A2 = A_arr @ A_arr
    A4 = A2 @ A2
    A6 = A4 @ A2
    A8 = A6 @ A2

    # make identity matrix of correct shape
    ident = tf.eye(
                num_rows    = tf.shape(A_arr)[-1],
                batch_shape = tf.shape(A_arr)[:-2],
                dtype       = A_arr.dtype)

    # scale the powers
    B  = A_arr  * 2**-s
    B2 = A2 * 2**(-2*s)
    B4 = A4 * 2**(-4*s)
    B6 = A6 * 2**(-6*s)

    # make U, V
    U2 = B6 @ (b[13]*B6 + b[11]*B4 + b[9]*B2)
    V2 = B6 @ (b[12]*B6 + b[10]*B4 + b[8]*B2)
    U  = B @ (U2 + b[7]*B6 + b[5]*B4 + b[3]*B2 + b[1]*ident)
    V  =      V2 + b[6]*B6 + b[4]*B4 + b[2]*B2 + b[0]*ident

    # solve Px = Q
    X = tf.linalg.solve(V-U, V+U)
    
    # square s times
    for i in range(s):
        X = X @ X
    
    return X


def expm_arr(A_arr, s):
    # calculate powers of A
    A2 = A_arr @ A_arr
    A4 = A2 @ A2
    A6 = A4 @ A2
    A8 = A6 @ A2
    ident = np.eye(A_arr.shape[-2], A_arr.shape[-1], dtype=A_arr.dtype)
    
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
    
    U  = B @ (U2 + b[7]*B6 + b[5]*B4 + b[3]*B2 + b[1]*ident[np.newaxis,:,:])
    V  =      V2 + b[6]*B6 + b[4]*B4 + b[2]*B2 + b[0]*ident[np.newaxis,:,:]
        
    # solve Px = Q
    X = np.linalg.solve(V-U, V+U)
    
    # square s times
    for i in range(s):
        X = X @ X
    
    return X

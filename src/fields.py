import numpy as np
import tensorflow as tf

def field(t, DCi, DCslope, ACi, deltaT, ACw, **kwargs):
    Ez = DCi - DCslope*t + ACi * (tf.math.sign(DCi/DCslope+deltaT - t)+1)/2 * tf.math.cos(ACw*t)
    return tf.transpose([0*Ez, 0*Ez, Ez, 0*Ez, 0*Ez, 0*Ez+0.5])

def time_mesh(DCi, DCslope, ACi, deltaT, ACw, pts_per_Vcm, num_segm, scan_length, **kwargs):
    # split the time period into a number of segments
    segm = np.linspace(0, scan_length*DCi/DCslope, int(3*DCi/DCslope*ACw/(2*np.pi)/num_segm))

    # make sure there's at least one segment
    if len(segm) < 2:
       segm = [0, scan_length*DCi/DCslope]
    
    # make a sub-mesh for each of the segments
    time = []
    for i in range(len(segm)-1):
        AC = ACi * (tf.math.sign(DCi/DCslope+deltaT - segm[i])+1)/2
        N_pts = (DCslope + ACw*AC) * (segm[i+1]-segm[i]) * pts_per_Vcm
        time.extend( np.linspace(segm[i], segm[i+1], int(N_pts)) )
        
    return np.array(time)


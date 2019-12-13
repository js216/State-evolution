import numpy as np

def DC(t, DCi, DCslope):
    return DCi - DCslope*t

def AC(t, ACi, T0):
    return ACi * np.heaviside( T0 - t, 1)

def Ez(t, DCi, DCslope, ACi, deltaT, ACw, **kwargs):
    return DC(t, DCi, DCslope) + AC(t, ACi, DCi/DCslope+deltaT) * np.cos(ACw*t)

def field(t, **params):
    field_arr = np.zeros([len(t), 6])
    field_arr[:,2] = Ez(t, **params)
    field_arr[:,5] = 0.5
    return field_arr

def time_mesh(DCi, DCslope, ACi, deltaT, ACw, pts_per_Vcm, num_segm, scan_len, **kwargs):
    # split the time period into a number of segments
    segm = np.linspace(0, scan_len*DCi/DCslope, int(3*DCi/DCslope*ACw/(2*np.pi)/num_segm))

    # make sure there's at least one segment
    if len(segm) < 2:
       segm = [0, scan_len*DCi/DCslope]
    
    # make a sub-mesh for each of the segments
    time = []
    for i in range(len(segm)-1):
        N_pts = (DCslope + ACw*AC(segm[i], ACi, DCi/DCslope+deltaT)) * (segm[i+1]-segm[i]) * pts_per_Vcm
        time.extend( np.linspace(segm[i], segm[i+1], int(N_pts)) )
        
    return np.array(time)


import numpy as np

def DC(t, DCi, DCslope):
    return DCi - DCslope*t

def AC(t, ACi, T0):
    return ACi * np.heaviside( T0 - t, 1)

def Ez(t, DCi, DCslope, ACi, deltaT, ACw):
    return DC(t, DCi, DCslope) + AC(t, ACi, DCi/DCslope+deltaT) * np.cos(ACw*t)

def field(t, **params):
    return np.array([[0,0,Ez(t, **params),0,0,0.5]])

def time_mesh(DCi, DCslope, ACi, deltaT, ACw, pts_per_Vcm, num_segm, scan_length):
    # split the time period into a number of segments
    segm = np.linspace(0, scan_length*DCi/DCslope, int(3*DCi/DCslope*ACw/(2*np.pi)/num_segm))

    # make sure there's at least one segment
    if len(segm) < 2:
       segm = [0, scan_length*DCi/DCslope]
    
    # make a sub-mesh for each of the segments
    time = []
    for i in range(len(segm)-1):
        N_pts = (DCslope + ACw*AC(segm[i], ACi, DCi/DCslope+deltaT)) * (segm[i+1]-segm[i]) * pts_per_Vcm
        time.extend( np.linspace(segm[i], segm[i+1], int(N_pts)) )
        
    return np.array(time)


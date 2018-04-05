import numpy as np
import pandas as pd

def rotate (origin, points, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    
    try:
        px, py = points[:,0], points[:,1]
    except TypeError:
        px, py = points[0], points[1]
        
    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    
    return qx, qy

def procrustes(p, angle, ma_window=12):
    """
    Normalize a set of 2D landmarks by translating, scaling and rotating.

    Set p of n-points expected to be n-by-2. Angle should be given in radians.

    """

    # Smooth by moving average if a window is provided
    if ma_window:
        
        s = pd.rolling_mean(p, ma_window)
        s[0:12, :] = p[0:12, :]
        p = s

    # Translate shape to center at origin by subtracting the mean of each shape
    p -= np.mean(p, axis=0)

    # Scaling by root mean squared of shape
    p /= np.sum(np.power(p, 2))
    
    # Rotation by provided angle
    px, py = rotate( (0, 0), p, angle)

    return np.vstack((px, py))

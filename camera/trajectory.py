import numpy as np


def trajectory_linear(start, stop, n_pic, offset=None):
    if offset is None:
        offset = [0, 0, 0]
    start = np.asarray(start, dtype=float).ravel()
    stop = np.asarray(stop, dtype=float).ravel()
    offset = np.asarray(offset, dtype=float).ravel()
    [x, y, z], [x_stop, y_stop, z_stop] = start + offset, stop + offset
    return np.mgrid[x:x_stop:n_pic * 1j, y:y_stop:n_pic * 1j, z:z_stop:n_pic * 1j].T


def trajectory_circle(r, npic_circle, npic, theta0=0, elevation=0, center=None):
    if center is None:
        center = [0, 0, 0]
    center = np.asarray(center, dtype=float).ravel()
    [x, y, z] = center
    theta = np.linspace(0, npic * 2 * np.pi / npic_circle, num=npic, endpoint=False)
    return np.asarray(
        [r * np.cos(theta + theta0) + x, r * np.sin(theta + theta0) + y, theta / (2 * np.pi) * elevation + z]).T

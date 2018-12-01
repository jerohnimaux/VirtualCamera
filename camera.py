import numpy as np


def intrinsic_from_params(f, theta, x, y):
    ratio = float(x) / y  # screen ratio
    theta_x = theta / (2 * 180) * np.pi  # angle in x of a pixel
    theta_y = theta_x / ratio  # angle in y of a pixel
    pixel_size_x = np.tan(theta_x) * f / (x / 2)
    pixel_size_y = np.tan(theta_y) * f / (y / 2)
    o_x, o_y = float(x / 2), float(y / 2)  # center of the image

    return np.asarray([[-f / pixel_size_x, 0, o_x], [0, -f / pixel_size_y, o_y], [0, 0, 1]], dtype=float)


def rotation_from_coord(coord, target=None):
    if target is None:
        target = [0, 0, 0]
    target = np.asarray(target, dtype=float).reshape(3)
    x, y, z = coord[:, 0] - target[0], coord[:, 1] - target[1], coord[:, 2] - target[2]
    theta_z = np.arctan2(y, x)
    theta_x = np.arctan2(z, np.sqrt(np.power(x, 2) + np.power(y, 2)))
    rz = np.around([[[np.cos(t + np.pi / 2), -np.sin(t + np.pi / 2), 0],  # rotation around cam z to align cam y-axis...
                     [np.sin(t + np.pi / 2), np.cos(t + np.pi / 2), 0],  # ... with world z-axis/camera center plan
                     [0, 0, 1.]] for t in theta_z], 2)
    rx = np.around([[[1, 0, 0],  # rotation around camera x to point cam z-axis...
                     [0, np.cos(np.pi / 2 + t), np.sin(np.pi / 2 + t)],  # to world center (target)
                     [0, -np.sin(np.pi / 2 + t), np.cos(np.pi / 2 + t)]] for t in theta_x], 2)

    return np.matmul(rx, rz)


def pose_from_coord(coord, target):
    return np.hstack((rotation_from_coord(coord, target).reshape(-1, 3), coord.reshape(-1, 1))).reshape(-1, 3, 4)


def extrinsic_from_pose(pose):
    rt = pose[:, :, 0:3].transpose(0, 2, 1)
    t = pose[:, :, 3].reshape(-1, 3, 1)
    rtt = np.matmul(rt, t)

    return np.concatenate((rt, -rtt), axis=2)


def projection_from_params(m_int, m_ext):
    return np.swapaxes(np.dot(m_int.reshape(3, 3), m_ext), 0, 1)


class CameraMatrix:
    """ Compute the intrinsic and extrinsic matrix of a virtual camera with no distortion from parameters.
    The extrinsic matrix is calculated from the coordinates of the camera in the world space of reference.
    trajectory_linear() and trajectory_circle() are methods to generate a list of camera coordinates and then generate the camera matrices.
    Intrinsic and extrinsic matrices are generated on call. Projection matrix is generated each time the coordinates are
    changed.

    Note : projection and extrinsic matrices are organised in a 3-dim numpy array, the first dimension is the list of
    all projection/extrinsic matrices (for every position of the camera).

    Example of use (with default intrinsic parameters):

        cam = CameraMatrix()
        cam.trajectory_circle(2,36,36)
        projection_matrices = cam.projection
        projection_matrix_for_position_4 = projection_matrices[3,:,:]

    """

    class Res:
        def __init__(self, x, y):
            """ Inner class to describe camera resolution with x and y parameters"""
            self.x = x
            self.y = y

        def tuple(self):
            return self.x, self.y

    def __init__(self, f=18, theta=60, res_x=640, res_y=480):
        self.f = f  # focal length of the camera in mm
        self.theta = theta  # FOV angle
        self.res = self.Res(res_x, res_y)  # resolution of the camera

        self.__coord = None
        self.target = None
        self.projection = None

    def __repr__(self):
        int_str = "Camera Intrinsic parameters :\n" \
                  "f = {}\n" \
                  "theta = {}\n" \
                  "resolution = {}x{}\n\n" \
                  "intrinsic matrix = \n" \
                  "{}\n".format(self.f, self.theta, self.res.x, self.res.y, self.intrinsic)
        ext_str = "extrinsic matrix = \n" \
                  "{}\n".format(self.intrinsic) if self.coord is not None else "extrinsic cannot be calculated\n"
        coord_str = "coordinates = \n" \
                    "{}\n".format(self.coord)
        proj_str = "projection matrix = \n" \
                   "{}\n".format(self.projection)

        return int_str + "\n" + ext_str + "\n" + coord_str + "\n" + proj_str

    @property
    def intrinsic(self):
        return intrinsic_from_params(self.f, self.theta, self.res.x, self.res.y)

    @property
    def extrinsic(self):
        if self.coord is not None:
            return extrinsic_from_pose(pose_from_coord(self.coord, self.target))
        else:
            print("Coordinates have not been calculated yet!")
            return None

    @property
    def coord(self):
        return self.__coord

    @coord.setter
    def coord(self, coord):
        self.__coord = np.asarray(coord, dtype=float).reshape(-1, 3)
        self.__compute_projection()

    def trajectory_linear(self, start, stop, n_pic, offset=None, target=None):
        if target is None:
            target = [0, 0, 0]
        if offset is None:
            offset = [0, 0, 0]
        start = np.asarray(start, dtype=float).ravel()
        stop = np.asarray(stop, dtype=float).ravel()
        offset = np.asarray(offset, dtype=float).ravel()
        target = np.asarray(target, dtype=float).ravel()
        self.target = target
        [x, y, z], [x_stop, y_stop, z_stop] = start + offset, stop + offset
        self.coord = np.mgrid[x:x_stop:n_pic * 1j, y:y_stop:n_pic * 1j, z:z_stop:n_pic * 1j].T
        return self.coord

    def trajectory_circle(self, r, npic_circle, npic, theta0=0, elevation=0, center=None, target=None):
        if target is None:
            target = [0, 0, 0]
        if center is None:
            center = [0, 0, 0]
        center = np.asarray(center, dtype=float).ravel()
        target = np.asarray(target, dtype=float).ravel()
        self.target = target
        [x, y, z] = center
        theta = np.linspace(0, npic * 2 * np.pi / npic_circle, num=npic, endpoint=False)
        self.coord = np.asarray(
            [r * np.cos(theta + theta0) + x, r * np.sin(theta + theta0) + y, theta / (2 * np.pi) * elevation] + z).T
        return self.coord

    def __compute_projection(self):
        try:
            self.projection = projection_from_params(self.intrinsic, self.extrinsic)
        except ValueError:
            print("Projection cannot be calculated!")
            self.__projection = None

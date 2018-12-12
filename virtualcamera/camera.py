import numpy as np


class Camera:
    """ Compute the intrinsic and extrinsic matrix of a virtual virtualcamera with no distortion from parameters.
    The extrinsic matrix is calculated from the coordinates of the virtualcamera in the world space of reference.
    trajectory_linear() and trajectory_circle() are methods to generate a list of virtualcamera coordinates and then generate
    the virtualcamera matrices.
    Intrinsic and extrinsic matrices are generated on call. Projection matrix is generated each time the coordinates are
    changed.

    Note : projection and extrinsic matrices are organised in a 3-dim numpy array, the first dimension is the list of
    all projection/extrinsic matrices (for every position of the virtualcamera).

    Example of use (with default intrinsic parameters):

        cam = Camera()
        cam.trajectory_circle(2,36,36)
        projection_matrices = cam.projection
        projection_matrix_for_4th_position = projection_matrices[3]

    """

    class Res:
        """ Inner class to describe virtualcamera resolution with x and y parameters"""

        def __init__(self, x, y):
            self.x = x
            self.y = y

        def tuple(self):
            return self.x, self.y

    def __init__(self, f=18, theta=60, res_x=640, res_y=480):
        self.f = f  # focal length of the virtualcamera in mm
        self.theta = theta  # FOV angle
        self.res = self.Res(res_x, res_y)  # resolution of the virtualcamera
        self.intrinsic = intrinsic_from_params(self.f, self.theta, self.res.x, self.res.y)

        self.__coord = None
        self.__target = None
        self.projection = None
        self.extrinsic = None

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
    def target(self):
        return self.__target

    @target.setter
    def target(self, coord):
        self.__target = np.asarray(coord, dtype=float).reshape(1, 3)
        if self.__coord is not None:
            self.__compute_projection()

    @property
    def coord(self):
        return self.__coord

    @coord.setter
    def coord(self, coord):
        self.__coord = np.asarray(coord, dtype=float).reshape(-1, 3)
        self.__compute_projection()

    def __compute_projection(self):
        try:
            self.intrinsic = intrinsic_from_params(self.f, self.theta, self.res.x, self.res.y)
            self.extrinsic = extrinsic_from_coord(self.coord, self.target)
            self.projection = np.matmul(self.intrinsic.reshape(3, 3), self.extrinsic)
        except ValueError:
            print("Projection cannot be calculated!")
            self.extrinsic = None
            self.projection = None

    def get_images(self, objects, fast=1):
            nimage = np.size(self.extrinsic, axis=0)
            image = np.zeros([nimage * self.res.y, self.res.x, 3], dtype=np.uint8)
            random_picking = np.random.choice(objects.shape[0], int(self.res.x * self.res.y * fast))
            objects = objects[random_picking, :]
            objects = np.hstack((objects, np.ones([np.size(objects, axis=0), 1])))
            pixels = np.matmul(objects, self.projection.transpose(0,2,1))  # (A * B.T).T == B * A.T

            # divide x, y by z
            pixels[..., 0:2] = (pixels[..., 0:2] / pixels[..., 2].reshape(-1, 1)).astype(np.int32)
            # remove pixels outside image frame
            pixels = pixels[(pixels[..., 0] > -1) & (pixels[..., 0] < self.res.x)
                            & (pixels[..., 1] > -1) & (pixels[..., 1] < self.res.y)]

            # keep only the closest pixels for each pixel coordinate
            if pixels.size != 0:
                unique, ind = np.unique(pixels[..., 0:2], axis=0, return_index=True)
                print(unique, unique.shape)
                # create image
                image[1 - np.uint(unique[:, 1]), np.uint(unique[:, 0]), :] = [255, 255, 0]

            return image.reshape(-1, self.res.y, self.res.x, 3)


def intrinsic_from_params(f, theta, x, y):
    """ Generate virtualcamera intrinsic matrix from virtualcamera parameters
    @:param f = focal distance (in mm)
    @:param theta = field of vision (in degrees)
    @:param x, y = x and y number of pixels
    @:returns virtualcamera intrinsic matrix
    """
    ratio = float(x) / y  # screen ratio
    theta_x = theta / (2 * 180) * np.pi  # angle in x of a pixel
    theta_y = theta_x / ratio  # angle in y of a pixel
    pixel_size_x = np.tan(theta_x) * f / (x / 2)
    pixel_size_y = np.tan(theta_y) * f / (y / 2)
    o_x, o_y = float(x / 2), float(y / 2)  # center of the image

    return np.asarray([[-f / pixel_size_x, 0, o_x], [0, -f / pixel_size_y, o_y], [0, 0, 1]], dtype=float)


def rotation_to_target(target):
    """ Rotation to face target ie to point -z virtualcamera axis to target (or vector of targets),
    with virtualcamera +y facing upward.
    @:param target = point to face in virtualcamera reference coordinate (vector of targets or target)
    @:returns rotation matrix (or vector of rotation matrices).
    """
    target = np.asarray(target).reshape(-1, 3)
    x, y, z = target[:, 0], target[:, 1], target[:, 2]
    theta_z = np.arctan2(x, y)
    theta_x = np.arctan2(z, np.sqrt(np.power(x, 2) + np.power(y, 2)))

    rz = np.around([[[np.cos(t), -np.sin(t), 0],  # rotation around cam z to align cam y-axis...
                     [np.sin(t), np.cos(t), 0],  # ... with world z-axis/virtualcamera center plan
                     [0, 0, 1]] for t in theta_z], 3)
    rx = np.around([[[1, 0, 0],  # rotation around virtualcamera x to point cam z-axis...
                     [0, np.cos(t + np.pi / 2), np.sin(t + np.pi / 2)],  # to world center (target)
                     [0, -np.sin(t + np.pi / 2), np.cos(t + np.pi / 2)]] for t in theta_x], 3)

    return np.matmul(rx, rz)


def extrinsic_from_coord(coord, offset=None):
    """ Generate the virtualcamera extrinsic matrix from a virtualcamera's world coordinates (or vector of coordinates).
    @:param coord = virtualcamera coordinates, in world reference (or vector of coordinates)
    @:param offset = target coordinates the virtualcamera must point at in world reference (or vector of coordinates)
    @:returns the extrinsic matrix of the virtualcamera
    """
    if offset is None:
        offset = [0, 0, 0]
    offset = np.asarray(offset, dtype=float).reshape(1, 3)
    coord = np.asarray(coord).reshape(-1, 1, 3)

    rotation = rotation_to_target(np.add(-coord, offset))
    translation = -np.matmul(rotation, coord.transpose(0, 2, 1))

    return np.hstack((rotation.reshape(-1, 3), translation.reshape(-1, 1))).reshape(-1, 3, 4)


def get_image(intrinsic, coord, pointofview, res, objects, fast=None):
    image = np.zeros([res.y, res.x, 3], dtype=np.uint8)

    if fast is not None:
        random_picking = np.random.choice(objects.shape[0], int(res.x * res.y * fast))
        objects = objects[random_picking, :]

    objects = np.hstack((objects, np.ones([np.size(objects, axis=0), 1])))
    extrinsic = extrinsic_from_coord(coord, pointofview)
    projection = intrinsic.dot(extrinsic[0])
    pixels = objects.dot(projection.T)  # (A * B.T).T == B * A.T

    # remove pixels outside from the back
    pixels = pixels[(pixels[:, 2] < 1)]
    # divide x, y by z
    pixels[:, 0:2] = (pixels[:, 0:2] / pixels[:, 2].reshape(-1, 1)).astype(np.int32)
    # remove pixels outside image frame
    pixels = pixels[(pixels[:, 0] > -1) & (pixels[:, 0] < res.x)
    & (pixels[:, 1] > -1) & (pixels[:, 1] < res.y)]

    # keep only the closest pixels for each pixel coordinate
    if pixels.size != 0:
        unique, ind = np.unique(pixels[:, 0:2], axis=0, return_index=True)
        # create image
        image[1 - np.uint(unique[:, 1]), np.uint(unique[:, 0]), :] = [255, 255, 0]

    return image
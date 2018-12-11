import numpy as np
import time

class RgbColor:
    """Color object in RGB format that can be manipulated

    Ex:
    color = RgbColor((100,100,0))
    color = color * 2
    print(color.tuple())  ==> (200, 200, 0)
    print((color + 1000).tuple)) ==> (255, 255, 0)
    """

    def __init__(self, color):
        self.r, self.g, self.b = color

    @staticmethod
    def _uint8(v):
        return max(0, min(255, np.uint8(v)))

    def tuple(self):
        return self.r, self.g, self.b

    def __add__(self, other):
        assert type(other) is int
        return RgbColor((self._uint8(self.r + other), self._uint8(self.g + other), self._uint8(self.b + other)))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __mul__(self, other):
        return RgbColor((self._uint8(round(self.r * other)),
                         self._uint8(round(self.g * other)),
                         self._uint8(round(self.b * other))))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self.__mul__(1 / other)


class Colorscale:
    """ Generate a colorscale from min to max of the color value"""

    def __init__(self, color, min=0.2, max=1, multicolor=False):
        self.color = color
        self.min = min
        self.max = max
        self.multicolor = multicolor  # not used yet

    def getcolor(self, x=1):
        return self.color * (self.min + min(1, abs(x)) * (self.max - self.min))

    def __mul__(self, other):
        return self.getcolor(other)

    def __rmul__(self, other):
        return self.__mul__(other)


def ratatinator(Intrinsic, Extrinsic, Imsize, points, fast_factor=1.):
    '''
    take an intrinsic matrix a extrinsic matrix and create
    :param Intrinsic:(3x3) intrinsic matrix of a camera
    :param Extrinsic:(3X4) extrinsic matrix of a camera
    :param points:(3xN) points array placed in colums. rows = number of points
    :param Imsize:(tuple) imsize in pix
    :return:
    '''
    t = time.time()
    img = np.zeros((Imsize[1], Imsize[0], 3))
    points = np.transpose(points)
    print("ps" + str(points.shape))

    img_size = Imsize[0]*Imsize[1]
    random_picking = np.random.choice(points.shape[1], int(img_size * fast_factor))
    print("rands" + str(random_picking))
    points = points[:, random_picking]

    N = points.shape[1]
    homogenous_points = np.vstack((points, np.ones((1, N))))

    cam_basis_points = np.dot(Extrinsic, homogenous_points)
    pix_basis_points = np.dot(Intrinsic, cam_basis_points).astype(np.float32)
    pix_basis_points[0:2, :] = (pix_basis_points[0:2, :] / pix_basis_points[2, :]).astype(np.int32)
    print(pix_basis_points.shape)

    pix_basis_points = pix_basis_points[:, (pix_basis_points[0, :] > -1) & (pix_basis_points[0, :] < Imsize[0]) & (pix_basis_points[1, :] > -1) & (pix_basis_points[1, :] < Imsize[1])]

    dist = np.sqrt(np.sum(cam_basis_points**2, axis=0))
    #print(dist)
    sorted_dist = np.flip(np.argsort(dist))

    dist_min = min(dist)
    dist_max = max(dist)
    alpha = 1/(dist_max-dist_min)

    elapsed1 = time.time() - t

    print("elaspsed1 : " + str(elapsed1))

    #-120 + (alpha * (dist[idx]-dist_min)) * 120
    for idx in sorted_dist:
        try:
            img[Imsize[1]-int(pix_basis_points[1, idx]), int(pix_basis_points[0, idx]), :] = \
                [np.uint8(255*(1-(alpha * (dist[idx]-dist_min)))), np.uint8(255*(1-(alpha * (dist[idx]-dist_min)))), np.uint8(0)]
        except IndexError:
            continue
    elapsed2 = time.time() - t
    print("elaspsed2 : " + str(elapsed2))
    return img
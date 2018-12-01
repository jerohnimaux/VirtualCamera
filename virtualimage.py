import numpy as np


class RgbColor:
    def __init__(self, color):
        self.r, self.g, self.b = color

    @staticmethod
    def _uint8(v):
        return max(0, min(255, int(v)))

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
    def __init__(self, color=(0, 255, 255), min=0.2, max=1, multicolor=False):
        self.color = RgbColor(color)
        self.min = min
        self.max = max
        self.multicolor = multicolor  # not used yet

    def getcolor(self, x=1):
        col = self.color * (self.min + min(1, abs(x)) * (self.max - self.min))
        return [col.r, col.g, col.b]

    def __mul__(self, other):
        return self.getcolor(other)

    def __rmul__(self, other):
        return self.__mul__(other)


def distances(coord, model):
    return np.linalg.norm((model - coord), axis=1)


def norm_distances(coord, model):
    d = distances(coord, model)
    return d / max(d)


def create_image(x, y, projection, model, distance, colorscale):
    image = np.zeros([y, x, 3], dtype=np.uint8)
    model = np.hstack((model, np.ones([np.size(model, axis=0), 1])))
    points = np.matmul(projection, model.T).T
    points = np.round(np.divide(points, points[:, 2].reshape([-1, 1]))).astype(np.uint)
    points = np.hstack((points[:, 0:2], distance.reshape(-1, 1)))

    uniques = np.unique(points[:, 0:2], axis=0)
    uniques = uniques[(np.where((0 <= uniques[:, 0]) &
                              (uniques[:, 0] < x) &
                              (0 <= uniques[:, 1]) &
                              (uniques[:, 1] < y)))]
    for row in uniques:
        image[uniques[1], uniques[0]] = colorscale * min(points[(np.where((points[:,0:2] == [uniques[0], uniques[1]])))][:, 2])

    return image

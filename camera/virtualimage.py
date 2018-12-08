import numpy as np


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
        return max(0, min(255, int(v)))

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


def create_image(x, y, projection, objects, colorscale):
    image = np.zeros([y, x, 3], dtype=np.uint8)
    objects = np.hstack((objects, np.ones([np.size(objects, axis=0), 1])))
    pixels = objects.dot(projection.T)  # (A * B.T).T == B * A.T

    # divide x, y by z
    pixels[:, 0:2] = (pixels[:, 0:2] / pixels[:, 2].reshape(-1, 1)).astype(np.int32)
    # remove pixels outside image frame
    pixels = pixels[(pixels[:, 0] > -1) & (pixels[:, 0] < x) & (pixels[:, 1] > -1) & (pixels[:, 1] < y)]

    # keep only the closest pixels for each pixel coordinate
    dist = np.linalg.norm(pixels, axis=1)  # calculate dist to objects
    pixels[:, 2] = dist / np.max(dist)  # range from 0 to 1 from 0 distance to farest object
    # pixels[:, 2] = (dist - min(dist)) / (max(dist) - min(dist))   # range from 0 to 1 from closest to farest object
    pixels = pixels[pixels[:, 2].argsort()]
    if pixels.size != 0:
        unique, ind = np.unique(pixels[:, 0:2], axis=0, return_index=True)
        pixels = pixels[ind]

    # generate image
    # y-coord is inversed (by convention)
    for pt in pixels:
        image[y - 1 - int(pt[1]), int(pt[0]), :] = (colorscale * (1 - pt[2])).tuple()

    return image

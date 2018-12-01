import virtualimage as vi
import camera as cm
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    cam = cm.CameraMatrix()
    model = np.load("cathedral_3dpts.npy")
    x_min, x_max = np.min(model[:,0]), np.max(model[:,0])
    y_min, y_max = np.min(model[:,1]), np.max(model[:,1])
    z_min, z_max = np.min(model[:,2]), np.max(model[:,2])
    target = center = [ (x_min + x_max ) /2, (y_min + y_max ) /2,(z_min + z_max ) /2 ]
    r = 2 * max( np.abs([x_max - x_min, y_max - y_min, z_max - z_min]))

    cam.trajectory_circle(r, 1, 2, center=center, target=target)

    color = vi.Colorscale(min=0.2, color=(255,255,0))
    images = []
    for coord, projection in zip(cam.coord, cam.projection):
        distance = vi.norm_distances(coord, model)
        images.append(vi.create_image(cam.res.x, cam.res.y, projection, model, distance, color))



import virtualcamera as cm
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    cam = cm.Camera()
    model = np.load("cathedral_3dpts.npy")

    x_min, x_max = np.min(model[:, 0]), np.max(model[:, 0])
    y_min, y_max = np.min(model[:, 1]), np.max(model[:, 1])
    z_min, z_max = np.min(model[:, 2]), np.max(model[:, 2])
    target = [(x_min + x_max) / 2, (y_min + y_max) / 2, 100]
    center = target
    center[2] = 0   # set center of rotation on the ground level
    r = 0.5 * max(np.abs([x_max - x_min, y_max - y_min, z_max - z_min]))

    cam.coord = cm.trajectory_circle(2 * r, 10, 20, center=center, elevation=100)
    cam.target = target
    color = cm.Colorscale(min=0.2, color=cm.RgbColor((0, 255, 255)))
    images = []
    print(cam)
    for coord, projection in zip(cam.coord, cam.projection):
        images.append(cm.create_image(cam.res.x, cam.res.y, projection, model, color))

    for image in images:
        img = plt.imshow(image)
        plt.show()

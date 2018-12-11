import virtualcamera as cm
import numpy as np

if __name__ == "__main__":

    # model = np.load("cathedral_3dpts.npy")
    #
    # x_min, x_max = np.min(model[:, 0]), np.max(model[:, 0])
    # y_min, y_max = np.min(model[:, 1]), np.max(model[:, 1])
    # z_min, z_max = np.min(model[:, 2]), np.max(model[:, 2])
    # target = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
    # center = target
    # center[2] = 0   # set center of rotation on the ground level
    # r = 0.5 * max(np.abs([x_max - x_min, y_max - y_min, z_max - z_min]))
    #
    # cam = cm.Camera(res_x=1024, res_y=768, theta=70)
    # cam.coord = cm.trajectory_circle(2 * r, 1, 1, center=center, elevation=50)
    # cam.target = target
    # color = cm.Colorscale(min=0.2, color=cm.RgbColor((0, 255, 255)))
    # images = []
    #
    # plt.imshow(cam.get_image(model)[0])
    # plt.show()
    # # for projection in cam.projection:
    # #     images.append(cm.create_image(cam.res.x, cam.res.y, projection, model, color))
    # #     #images.append(cm.ratatinator(cam.intrinsic, cam.extrinsic[0], (cam.res.x, cam.res.y), model))
    # # for image in images:
    # #     print(image.shape)
    # #     plt.imshow(image)
    # #     plt.show()

    model = np.load("cathe3D.npy")
    x_min, x_max = np.min(model[:, 0]), np.max(model[:, 0])
    y_min, y_max = np.min(model[:, 1]), np.max(model[:, 1])
    z_min, z_max = np.min(model[:, 2]), np.max(model[:, 2])
    target = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
    center = target
    center[2] = 0   # set center of rotation on the ground level
    r = 0.5 * max(np.abs([x_max - x_min, y_max - y_min, z_max - z_min]))
    print("center = ", center)
    game = cm.FPS(model, 10)
    game.start()

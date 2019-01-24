from .camera import Camera, get_image
import numpy as np
import pygame
from pygame.locals import *

class FPS:
    def __init__(self, objects, move_step=1, v_angle_step=np.pi / 12, h_angle_step=np.pi / 12, res_x=640, res_y=480,
                 fov=60, fast=1):
        """ This object uses Pygame functionnality to display images of the view of a 3D object from a virtual camera.
        The keyboard is used to move the camera in the 3D world, and the images are updated.
        @:param objects = array of the 3D points of the object in world coordinates
        @:param move_step = size of 1 step of movement (front, back, strafe left, strafe right)
        @:param v_angle_step = vertical angle step, the angle (in rad) of which the view is rotated when looking
        up or down.
        @:param h_angle_step = the angle (in rad) of which the view is rotated when looking left or right.
        @:param res_x = horizontal resolution
        @:param res_y = vertical resolution
        @:param fov = field of view (in degree)
        @:param fast = factor to improve performance. Decreases the number of 3D points to take into account.
                if set to None, then all points are used to generate the image."""

        self.run = True
        self.move_step = move_step
        self.v_angle_step = v_angle_step
        self.h_angle_step = h_angle_step
        self.commands = {K_z: "front",
                         K_s: "back",
                         K_q: "strafe_left",
                         K_d: "strafe_right",
                         K_UP: "look_up",
                         K_DOWN: "look_down",
                         K_LEFT: "look_left",
                         K_RIGHT: "look_right",
                         K_SPACE: "flatten",
                         K_x: "exit"
                         }

        self.pos = np.asarray([0, 0, 0], dtype=float)
        self.orientation = np.asarray([1, 0, 0], dtype=float)
        self.view = Camera(f=fov, res_x=res_x, res_y=res_y)
        self.objects = objects
        self.fast = fast

        pygame.init()
        self.fenetre = pygame.display.set_mode((self.view.res.x, self.view.res.y))

    @staticmethod
    def rx(theta):
        return np.array([[1, 0, 0],
                         [0, np.cos(theta), -np.sin(theta)],
                         [0, np.sin(theta), np.cos(theta)]])

    @staticmethod
    def rz(theta):
        return np.array([[np.cos(theta), -np.sin(theta), 0],
                         [np.sin(theta), np.cos(theta), 0],
                         [0, 0, 1]])

    @property
    def target(self):
        return self.pos + self.orientation

    def display(self):
        """ Update the image"""
        img = get_image(self.view.intrinsic, self.pos, self.target, self.view.res, self.objects, self.fast).transpose(1, 0, 2)
        surface = pygame.surfarray.make_surface(img)
        self.fenetre.blit(surface, (0, 0))
        pygame.display.flip()

    def start(self):
        """Start the game"""
        self.display()
        self.run = True

        while self.run:
            for event in pygame.event.get():  # Attente des événements
                if event.type == KEYDOWN:
                    try:
                        getattr(self, self.commands[event.key])() # Execute la méthode en fonction de la commande clavier
                    except KeyError:
                        continue
            # Rafraichissement
            self.display()

    def front(self):
        self.pos += self.orientation * self.move_step

    def back(self):
        self.pos -= self.orientation * self.move_step

    def strafe_left(self):
        side2D = self.orientation[0:2] * self.move_step
        self.pos = np.asarray([self.pos[0] - side2D[1], self.pos[1] + side2D[0], self.pos[2]])

    def strafe_right(self):
        side2D = self.orientation[0:2] * self.move_step
        self.pos = np.asarray([self.pos[0] + side2D[1], self.pos[1] - side2D[0], self.pos[2]])

    def look_up(self):
        z = np.arctan2(self.orientation[0], self.orientation[1])
        max_x = np.arctan2(np.sqrt(self.orientation[0] ** 2 + self.orientation[1] ** 2), self.orientation[2]) - 0.1
        print("max_x_up =", max_x, " theta = ", self.h_angle_step)
        self.orientation = np.round(
            self.rz(-z).dot(self.rx(min(max_x, self.h_angle_step)).dot(self.rz(z).dot(self.orientation))), 4)
        print("pos = ", self.pos, "target = ", self.target)

    def look_down(self):
        z = np.arctan2(self.orientation[0], self.orientation[1])
        max_x = np.arctan(
            np.sqrt(self.orientation[0] ** 2 + self.orientation[1] ** 2) / max(0.01, -self.orientation[2])) - 0.1
        print("max_x_down =", max_x, " theta = ", self.h_angle_step)

        self.orientation = np.round(
            self.rz(-z).dot(self.rx(-min(max_x, self.h_angle_step)).dot(self.rz(z).dot(self.orientation))), 4)
        print("pos = ", self.pos, "target = ", self.target)

    def look_left(self):
        self.orientation = self.rz(self.h_angle_step).dot(self.orientation)

    def look_right(self):
        self.orientation = self.rz(-self.h_angle_step).dot(self.orientation)

    def flatten(self):
        self.orientation[2] = 0
        self.orientation *= np.sum(self.orientation)

    def exit(self):
        self.run = False

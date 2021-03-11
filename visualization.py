"""
Author: Amr Elsersy
email: amrelsersay@gmail.com
-----------------------------------------------------------------------------------
Description: Visualization of dataset with annotations in cv2 & tensorboard
"""

import numpy as np
import cv2
import argparse
from dataset import WFLW_Dataset
from dataset import create_train_loader, create_test_loader

import torch
from torchvision.utils import make_grid
import torch.utils.tensorboard as tensorboard

from utils import *

class WFLW_Visualizer:
    def __init__(self):
        self.writer = tensorboard.SummaryWriter("checkpoint/tensorboard")

        self.rect_color = (0,255,255)
        self.landmarks_color  = (0,255,0)
        self.rect_width = 3
        self.landmarks_radius = 1
        self.winname = "image"
        self.crop_resize_shape = (400, 400)
        self.user_press = None

    def visualize(self, image, labels, draw_eulers = False):
        landmarks = labels['landmarks'].astype(np.int32)
        euler_angles = labels['euler_angles']
        
        image = self.draw_landmarks(image, landmarks)
        if draw_eulers:
            image = self.draw_euler_angles_approximation(image, euler_angles)
        self.show(image)        

    def show(self, image, size = None, wait = True, winname="image"):
        if size:
            image = cv2.resize(image, size)
        else:
            image = cv2.resize(image, self.crop_resize_shape)

        cv2.imshow(winname, image)
        if wait:
            self.user_press = cv2.waitKey(0) & 0xff

    def draw_landmarks(self, image, landmarks):
        for (x,y) in landmarks:
            cv2.circle(image, (x,y), self.landmarks_radius, self.landmarks_color, -1)
        return image                

    def batch_draw_landmarks(self, images, labels):
        n_batches = images.shape[0]
        for i in range(n_batches):
            image = images[i]
            
            landmarks = labels['landmarks'].type(torch.IntTensor)
            landmarks = landmarks[i]

            image = self.draw_landmarks(image.numpy(), landmarks)
            images[i] = torch.from_numpy(image)

        return images

    def draw_euler_angles(self, image, rvec, tvec, euler_angles, intrensic_matrix):
        # i, j, k axes in world 3D coord.
        axis = np.identity(3) * 5
        # axis_img_pts = intrensic * exstrinsic * axis
        axis_pts = cv2.projectPoints(axis, rvec, tvec, intrensic_matrix, None)[0]
        image = self.draw_euler_axis(image, axis_pts, euler_angles)

        return image

    def draw_euler_angles_approximation(self, image, euler_angles):
        axis = np.identity(3) * 5

        rotation = euler_to_rotation(euler_angles)
        # for just visualization we will use the avarage value of tvec
        tvec = np.array([
            [-1],
            [-2],
            [-21]
        ], dtype=np.float)

        intrensic = get_intrensic_matrix(image)

        # from world space to 3D cam space
        axis_pts = rotation @ axis + tvec
        # project to image
        axis_pts = intrensic @ axis_pts
        # convert from homoginous to image plane
        axis_pts /= axis_pts[2]
        # don't need the z component
        axis_pts = np.delete(axis_pts, 2, axis=0).T

        image = self.draw_euler_axis(image, axis_pts, euler_angles)
        return image

    def draw_euler_axis(self, image, axis_pts, euler_angles):
        """
            draw euler axes in the image center 
        """
        center = (image.shape[1]//2, image.shape[0]//2)

        axis_pts = axis_pts.astype(np.int32)
        pitch_point = tuple(axis_pts[0].ravel())
        yaw_point   = tuple(axis_pts[1].ravel())
        roll_point  = tuple(axis_pts[2].ravel())

        pitch_color = (255,255,0)
        yaw_color   = (0,255,0)
        roll_color  = (0,0,255)

        pitch, yaw, roll = euler_angles

        cv2.line(image, center,  pitch_point, pitch_color, 2)
        cv2.line(image, center,  yaw_point, yaw_color, 2)
        cv2.line(image, center,  roll_point, roll_color, 2)
        cv2.putText(image, "Pitch:{:.2f}".format(pitch), (0,10), cv2.FONT_HERSHEY_PLAIN, 1, pitch_color)
        cv2.putText(image, "Yaw:{:.2f}".format(yaw), (0,20), cv2.FONT_HERSHEY_PLAIN, 1, yaw_color)
        cv2.putText(image, "Roll:{:.2f}".format(roll), (0,30), cv2.FONT_HERSHEY_PLAIN, 1, roll_color)

        # origin
        cv2.circle(image, center, 2, (255,255,255), -1)
        return image

    def visualize_tensorboard(self, images, labels, step=0):
        images = self.batch_draw_landmarks(images, labels)
        # format must be specified (N, H, W, C)
        self.writer.add_images("images", images, global_step=step, dataformats="NHW")



if __name__ == "__main__":
    # ======== Argparser ===========
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', choices=['train', 'test'], help="choose which dataset to visualize")
    parser.add_argument('--tensorboard', action='store_true', help="visualize images to tensorboard")
    parser.add_argument('--stop_batch', type=int, default=5, help="tensorboard batch index to stop")
    args = parser.parse_args()
    # ================================

    visualizer = WFLW_Visualizer()

    # Visualize the dataset (train or val) with landmarks
    if not args.tensorboard:
        dataset = WFLW_Dataset(mode=args.mode)
        for i in range(len(dataset)):
            image, labels = dataset[i]
            print('landmarks', labels['landmarks'])

            print ("*" * 80, '\n\n\t press n for next example .... ESC to exit')
            print('\tcurrent image: ',labels['image_name'])

            visualizer.visualize(image, labels)            
            if visualizer.user_press == 27:
                break
            

    # Tensorboard Visualization on 5 batches with 64 batch size
    else:
        batch_size = 64
        dataloader = create_test_loader(batch_size=batch_size, transform=None)

        batch = 0
        for (images, labels) in dataloader:
            batch += 1

            visualizer.visualize_tensorboard(images, labels, batch)
            print ("*" * 60, f'\n\n\t Saved {batch_size} images with Step{batch}. run tensorboard @ project root')
            
            if batch == args.stop_batch:
                break

        visualizer.writer.close()



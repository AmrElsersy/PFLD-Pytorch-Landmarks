
import numpy as np
import cv2
import argparse
from dataset import WFLW_Dataset, LoadMode
from dataloader import create_train_loader, create_test_loader

import torch
from torchvision.utils import make_grid
import torch.utils.tensorboard as tensorboard

class WFLW_Visualizer:
    def __init__(self, mode = LoadMode.FULL_IMG):
        self.mode = mode
        self.writer = tensorboard.SummaryWriter("tensorboard")

        self.rect_color = (0,255,255)
        self.landmarks_color  = (0,255,0)
        self.rect_width = 3
        self.landmarks_radius = 1
        self.winname = "image"
        self.full_resize_shape = (1000, 900)
        self.crop_resize_shape = (300, 400)
        self.user_press = None

    def visualize(self, image, labels):
        rect = labels['rect'].astype(np.int32)
        landmarks = labels['landmarks'].astype(np.int32)
        image = self.draw_landmarks(image, rect, landmarks)
        self.show(image)        

    def show(self, image):
        if self.mode == LoadMode.FULL_IMG:
            image = cv2.resize(image, self.full_resize_shape)
        elif self.mode == LoadMode.FACE_ONLY:
            image = cv2.resize(image, self.crop_resize_shape)

        cv2.imshow(self.winname, image)
        self.user_press = cv2.waitKey(0) & 0xff

    def draw_landmarks(self, image, rect, landmarks):
        if self.mode == LoadMode.FULL_IMG:
            (x1,y1), (x2,y2) = rect
            cv2.rectangle(image, (x1,y1), (x2,y2), self.rect_color, self.rect_width)

        for (x,y) in landmarks:
            cv2.circle(image, (x,y), self.landmarks_radius, self.landmarks_color, -1)

        return image                

    def batch_draw_landmarks(self, images, labels):
        n_batches = images.shape[0]
        for i in range(n_batches):
            image = images[i]
            
            rect = labels['rect'].type(torch.IntTensor)
            landmarks = labels['landmarks'].type(torch.IntTensor)

            rect = rect[i]
            landmarks = landmarks[i]

            image = self.draw_landmarks(image.numpy(), rect, landmarks)
            images[i] = torch.from_numpy(image)

        return images

    def visualize_tensorboard(self, images, labels, step=0):
        images = self.batch_draw_landmarks(images, labels)
        # format must be specified (N, H, W, C)
        self.writer.add_images("images", images, global_step=step, dataformats="NHWC")


if __name__ == "__main__":
    # ======== Argparser ===========
    parser = argparse.ArgumentParser()
    parser.add_argument('--full_img', action="store_true", help="full image with 1 face rect or only the face")
    parser.add_argument('--tensorboard', action='store_true', help="visualize images to tensorboard")
    args = parser.parse_args()
    # ================================

    mode = LoadMode.FACE_ONLY 
    mode = LoadMode.FULL_IMG if args.full_img else mode
    visualizer = WFLW_Visualizer(mode)

    # Visualize the dataset (train or val) with landmarks
    if not args.tensorboard:
        dataset = WFLW_Dataset(mode='train', load_mode=mode)
        for i in range(len(dataset)):
            image, labels = dataset[i]
            visualizer.visualize(image, labels)

            if visualizer.user_press == 27:
                break

            print ("*" * 80, '\n\n\t press n for next example .... ESC to exit')


    # Tensorboard Visualization on 5 batches with 64 batch size
    else:
        batch_size = 64
        dataloader = create_test_loader(batch_size=batch_size, mode=mode)

        batch = 0
        for (images, labels) in dataloader:
            batch += 1

            visualizer.visualize_tensorboard(images, labels, batch)
            print ("*" * 60, f'\n\n\t Saved {batch_size} images with Step{batch}. run tensorboard @ project root')
            
            if batch == 5:
                break

        visualizer.writer.close()



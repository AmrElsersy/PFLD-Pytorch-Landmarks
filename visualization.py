
import numpy as np
import cv2
import argparse
from dataset import WFLW_Dataset, LoadMode

class WFLW_Visualizer:
    def __init__(self, mode = LoadMode.FULL_IMG):
        self.rect_color = (0,255,255)
        self.rect_width = 3
        self.landmarks_color  = (0,255,0)
        self.landmarks_radius = 1
        self.winname = "image"
        self.full_resize_shape = (1000, 900)
        self.crop_resize_shape = (112, 112)
        self.user_press = None
        self.mode = mode

    def visualize(self, image, labels):
        rect = labels['rect'].astype(np.int32)
        landmarks = labels['landmarks'].astype(np.int32)
        image = self.draw_landmarks(image, rect, landmarks)
        self.show(image)

    def draw_landmarks(self, image, rect, landmarks):
        (x1,y1), (x2,y2) = rect
        cv2.rectangle(image, (x1,y1), (x2,y2), self.rect_color, self.rect_width)
        for point in landmarks:
            (x,y) = point
            cv2.circle(image, (x,y), self.landmarks_radius, self.landmarks_color, -1)
        return image                

    def show(self, image):
        if mode == LoadMode.FULL_IMG:
            image = cv2.resize(image, self.resize_shape)
        elif mode == LoadMode.FACE_ONLY:
            image = cv2.resize(image, self.crop_resize_shape)

        cv2.imshow(self.winname, image)
        self.user_press = cv2.waitKey(0) & 0xff


if __name__ == "__main__":
    # ======== Argparser ===========
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='full-img', choices=['full-img', 'face-img'], 
                        help="full image with 1 face rect or only the face")
    args = parser.parse_args()
    mode = None 
    if args.mode == 'full-img':
        mode = LoadMode.FULL_IMG
    elif args.mode == 'face-img':
        mode = LoadMode.FACE_ONLY
    # ================================

    dataset = WFLW_Dataset(mode='train', load_mode=mode)
    visualizer = WFLW_Visualizer(mode)

    for i in range(len(dataset)):
        image, labels = dataset[i]
        visualizer.visualize(image, labels)
        
        if visualizer.user_press == 27:
            break

        print ("*" * 40, '\n\n\t press n for next example .... ESC to exit')
        
        

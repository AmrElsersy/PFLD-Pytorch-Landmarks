"""
Author: Amr Elsersy
email: amrelsersay@gmail.com
-----------------------------------------------------------------------------------
Description: WFLW Dataset module to read images with annotations
"""

import os, time, enum
from PIL import Image
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np 
import cv2

class LoadMode(enum.Enum):
    FACE_ONLY = 0
    FULL_IMG = 1

class WFLW_Dataset(Dataset):
    def __init__(self, root='data/WFLW', mode='train', load_mode = LoadMode.FACE_ONLY):
        self.root = root
        self.images_root = os.path.join(self.root, "WFLW_images")
        self.face_shape = (112,112)

        self.mode = mode 
        self.load_mode = load_mode
        assert mode in ['train', 'val']

        self.annotations_root = os.path.join(self.root, "WFLW_annotations")
        self.train_test_root = os.path.join(self.annotations_root, "list_98pt_rect_attr_train_test")
        self.train_name = os.path.join(self.train_test_root, "list_98pt_rect_attr_train.txt")
        self.test_name = os.path.join(self.train_test_root, "list_98pt_rect_attr_test.txt")

        self.test_file  = open(self.test_name ,'r')
        self.train_file = open(self.train_name,'r')

        self.test_lines  = self.test_file.read().splitlines()
        self.train_lines = self.train_file.read().splitlines()

    def __getitem__(self, index):

        labels = self.read_annotations(index)
        image = self.read_image(labels['image_name'])

        # clip the image rect and translate the landmarks coordinates
        if self.load_mode == LoadMode.FACE_ONLY:
            rect = labels['rect']
            landmarks = labels['landmarks']

            # crop face
            (x1, y1), (x2, y2) = rect            
            image = image[int(y1):int(y2), int(x1):int(x2)]

            # resize the image & store the dims to resize landmarks
            h, w = image.shape[:2]
            image = cv2.resize(image, self.face_shape)
            new_h, new_w = self.face_shape

            # scale factor in x & y to scale the landmarks
            fx = new_w / w
            fy = new_h / h
            # translate the landmarks then scale them
            landmarks -= rect[0]
            for landmark in landmarks:
                landmark[0] *= fx
                landmark[1] *= fy

            # face rect
            rect[0] = (0,0)
            rect[1] = (x2-x1, y2-y1)
        
        return image, labels

    def read_annotations(self, index):
        annotations = self.train_lines[index] if self.mode == 'train' else self.test_lines[index]
        annotations = annotations.split(' ')
            
        # 98 lanamark points
        # pose expression illumination make-up occlusion blur
        # rect coordinates (x_min, y_min)(right-down) & (x_max, y_max)(top-left)
        # image sub-path that contains the face: sub_folder/image_name

        landmarks = annotations[0:196]
        rect = annotations[196:200]        
        attributes = annotations[200:206]
        image_name = annotations[206]

        # strings to num
        landmarks = [float(landmark) for landmark in landmarks]
        rect = [float(coord) for coord in rect]
        attributes = [int(attribute) for attribute in attributes]

        # list to numpy 
        landmarks = np.array(landmarks, dtype=np.float).reshape(98,2)
        rect = np.array(rect, dtype=np.float).reshape((2,2))
        attributes = np.array(attributes, dtype=np.int).reshape((6,))

        labels = {
            "landmarks" : landmarks,
            "rect"      : rect,
            "attributes": attributes,
            "image_name": image_name
        }

        return labels

    def read_image(self, name):
        path = os.path.join(self.images_root, name)
        # image = Image.open(path)
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        return image        


    def __len__(self):
        if self.mode == 'train':
            return len(self.train_lines)
        elif self.mode == 'val':
            return len(self.test_lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='val', choices=['train', 'val'])
    args = parser.parse_args()

    dataset = WFLW_Dataset(mode=args.mode)

    for i in range(len(dataset)):
        image, labels = dataset[i]

        print(labels)
        print('***' * 40, '\n')        
        time.sleep(2)



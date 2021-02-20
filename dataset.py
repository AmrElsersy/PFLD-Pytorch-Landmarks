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
import torchvision.transforms.transforms as transforms
import numpy as np 
import cv2

class WFLW_Dataset(Dataset):
    def __init__(self, root='data', mode='train', transform=None):
        self.root = root
        self.transform = transform

        self.mode = mode 
        assert mode in ['train', 'test']

        self.images_root = os.path.join(self.root, self.mode, "images")
        self.annotations_root = os.path.join(self.root, self.mode, "annotations.txt")

        self.annotations_file  = open(self.annotations_root ,'r')
        self.annotations_lines  = self.annotations_file.read().splitlines()

    def __getitem__(self, index):

        labels = self.read_annotations(index)
        image = self.read_image(labels['image_name'])
        labels['landmarks'] *= 112
        
        if self.transform:
            # to tensor
            # temp = np.copy(image)
            # temp = self.transform(temp)
            image = self.transform(image)
            # print('image after', image, '\n\n')
            # print('temp after', temp, '\n\n')

            # Noramlization Landmarks
            labels['landmarks'] = self.transform(labels['landmarks']) / 112
            # print(labels['landmarks'])

            # to tensor
            labels['attributes'] = self.transform(labels['attributes'].reshape(1,6))
            labels['euler_angles'] = self.transform(labels['euler_angles'].reshape(1,3))

        return image, labels

    def read_annotations(self, index):
        annotations = self.annotations_lines[index]
        annotations = annotations.split(' ')
            
        # 98 lanamark points
        # pose expression illumination make-up occlusion blur
        # image relative-path 

        image_name = annotations[0]
        landmarks = annotations[1:197]
        attributes = annotations[197:203]
        euler_angles = annotations[203:206]

        # strings to num
        landmarks = [float(landmark) for landmark in landmarks]
        attributes = [int(attribute) for attribute in attributes]
        euler_angles = [float(angle) for angle in euler_angles]
        
        # list to numpy 
        landmarks = np.array(landmarks, dtype=np.float).reshape(98,2)
        attributes = np.array(attributes, dtype=np.int).reshape((6,))
        euler_angles = np.array(euler_angles, dtype=np.float).reshape((3,))

        labels = {
            "landmarks" : landmarks,
            "attributes": attributes,
            "image_name": image_name,
            "euler_angles": euler_angles
        }

        return labels

    def read_image(self, path):
        # path = os.path.join(path)
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        return image        

    def __len__(self):
        return len(self.annotations_lines)


def create_train_loader(root='data', batch_size = 64, transform=transforms.ToTensor()):
    dataset = WFLW_Dataset(root, mode='train', transform=transform)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
    return dataloader

def create_test_loader(root='data', batch_size = 1, transform=transforms.ToTensor()):
    dataset = WFLW_Dataset(root, mode='test', transform=transform)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size)
    return dataloader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    args = parser.parse_args()

    dataset = WFLW_Dataset(mode=args.mode, transform= transforms.ToTensor())
    for i in range(len(dataset)):
        image, labels = dataset[i]
    
    # dataloader = create_train_loader(batch_size=1)
    # for image, labels in dataloader:
        
        print("image.shape",image.shape)
        print("landmarks.shape",labels['landmarks'])
        print("euler_angles.shape",labels['euler_angles'])
        print("attributes.shape",labels['attributes'])
        print('***' * 40, '\n')        
        

        time.sleep(1)

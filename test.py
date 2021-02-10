"""
Author: Amr Elsersy
email: amrelsersay@gmail.com
-----------------------------------------------------------------------------------
Description: Training & Validation
"""
import numpy as np 
import argparse
import logging
import time
import os
from tqdm import tqdm
import cv2

import torch
import torch.nn as nn
import torch.optim
import torch.utils.tensorboard as tensorboard

from dataset import WFLW_Dataset, LoadMode
from dataset import create_test_loader, create_train_loader
from visualization import WFLW_Visualizer

from model.Loss import PFLD_L2Loss
from model.model import PFLD, AuxiliaryNet
from model.DepthSepConv import DepthSepConvBlock
from model.BottleneckResidual import BottleneckResidualBlock

from utils import to_numpy_image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=20, help="training batch size")
    parser.add_argument('--tensorboard', type=str, default='checkpoint/tensorboard', help='path log dir of tensorboard')
    parser.add_argument('--datapath', type=str, default='data/WFLW', help='root path of WFLW dataset')
    parser.add_argument('--pretrained',type=str,default='checkpoint/model_weights/weights.pth.tar',help='load weights')
    args = parser.parse_args()
    return args

# ======================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = parse_args()
writer = tensorboard.SummaryWriter(args.tensorboard)

def main():
    # ========= dataset ===========
    dataset = WFLW_Dataset(root=args.datapath, mode='train', transform=True)
    visualizer = WFLW_Visualizer()
    # =========== models ============= 
    pfld = PFLD().to(device)
    auxiliarynet = AuxiliaryNet().to(device)
    # ========= load weights ===========
    checkpoint = torch.load(args.pretrained)
    pfld.load_state_dict(checkpoint["pfld"])
    auxiliarynet.load_state_dict(checkpoint["auxiliary"])
    print(f'\n\tLoaded checkpoint from {args.pretrained}\n')
    time.sleep(1)

    pfld.eval()
    auxiliarynet.eval()

    with torch.no_grad():
        for i in range(len(dataset)):
            image, labels = dataset[i]

            image = image.unsqueeze(0)
            landmarks = labels['landmarks'].squeeze() # shape (batch, 98, 2)
            # landmarks = landmarks.reshape((1, 196)) # reshape landmarks to match loss function

            # print('landmarks:',landmarks.shape)
            # print('attributes:',attributes.shape)
            # print('image:',image.shape)
            # print('euler:',euler_angles.shape)

            image = image.to(device)
            # landmarks = landmarks.to(device)

            pfld = pfld.to(device)
            auxiliarynet = auxiliarynet.to(device)

            featrues, pred_landmarks = pfld(image)
            pred_angles = auxiliarynet(featrues)
            pred_landmarks = pred_landmarks.cpu().reshape(98,2).numpy()
            image = to_numpy_image(image[0].cpu())

            pred_landmarks = (pred_landmarks*112).astype(np.int32) 
            print(pred_landmarks)
            print(landmarks)
            image = (image*255).astype(np.uint8)
            image = np.clip(image, 0, 255)

            cv2.imwrite("ray2.jpg", image)
            img = cv2.imread("ray2.jpg")
            img2 = np.copy(img)
            img2[:,:] = 0

            img = visualizer.draw_landmarks(img, pred_landmarks)
            # img2 = visualizer.draw_landmarks(img2, pred_landmarks)
            # cv2.imshow("RR",img2)

            visualizer.show(img)
            print('*'*70,'\n')

            if visualizer.user_press == 27:
                break

    writer.close()


if __name__ == "__main__":
    main()

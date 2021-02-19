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

from dataset import WFLW_Dataset
from dataset import create_test_loader, create_train_loader
from visualization import WFLW_Visualizer
import torchvision.transforms.transforms as transforms

from model.Loss import PFLD_L2Loss
from model.model import PFLD, AuxiliaryNet
from model.DepthSepConv import DepthSepConvBlock
from model.BottleneckResidual import BottleneckResidualBlock

from utils import to_numpy_image
import torch.backends.cudnn as cudnn

cudnn.benchmark = True
cudnn.determinstic = True
cudnn.enabled = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default='data', help='root path of augumented WFLW dataset')
    parser.add_argument('--pretrained',type=str,default='checkpoint/model_weights/weights.pth.tar',help='load weights')
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
    args = parser.parse_args()
    return args

# ======================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = parse_args()

def main():
    # ========= dataset ===========
    dataset = WFLW_Dataset(root=args.datapath, mode=args.mode, transform=transforms.ToTensor())
    visualizer = WFLW_Visualizer()
    # =========== models ============= 
    pfld = PFLD().to(device)
    auxiliarynet = AuxiliaryNet().to(device)
    print(pfld)
    # ========= load weights ===========
    checkpoint = torch.load(args.pretrained, map_location=device)
    # print(pfld.load_state_dict(checkpoint["pfld"]).keys())
    # return
    pfld.load_state_dict(checkpoint["pfld"], strict=False)
    auxiliarynet.load_state_dict(checkpoint["auxiliary"])
    print(f'\n\tLoaded checkpoint from {args.pretrained}\n')
    time.sleep(1)

    pfld.eval()
    auxiliarynet.eval()

    with torch.no_grad():
        for i in range(len(dataset)):
            image, labels = dataset[i]

            image = image.unsqueeze(0)
            image = image.to(device)
            landmarks = labels['landmarks'].squeeze() # shape (batch, 98, 2)

            pfld = pfld.to(device)
            auxiliarynet = auxiliarynet.to(device)
            t1 = time.time()
            featrues, pred_landmarks = pfld(image)
            t2 = time.time()
            pred_angles = auxiliarynet(featrues)
            print('pred_angles',pred_angles)
            print('gt_angles',labels['euler_angles'])
            print(f'gt_landmarks shape {landmarks.shape}, \n {landmarks[:5]}')
            print(f'landmarks shape {pred_landmarks.shape}, \n {pred_landmarks.reshape(98,2)[:5]}')

            t3 = time.time()
            # print(f"\ttime PFLD landmarks= {round((t2-t1)*1000,3)} ms")
            # print(f"\ttime auxiliary euler= {round((t3-t2)*1000,3)} ms")
            # print(f"\ttotal time= {round((t3-t1)*1000,3)} ms\n")

            landmarks = landmarks.cpu().reshape(98,2).numpy()
            landmarks = (landmarks*112.0).astype(np.int32) 

            pred_landmarks = pred_landmarks.cpu().reshape(98,2).numpy()
            pred_landmarks = (pred_landmarks*112.0).astype(np.int32) 

            image = to_numpy_image(image[0].cpu())
            image = (image*255).astype(np.uint8)
            image = np.clip(image, 0, 255)

            cv2.imwrite("ray2.jpg", image)
            img = cv2.imread("ray2.jpg")
            img2 = np.copy(img)
            img2[:,:] = 0

            visualizer.landmarks_radius = 1
            visualizer.landmarks_color = (0,255,0)
            img = visualizer.draw_landmarks(img, pred_landmarks)
            img2 = visualizer.draw_landmarks(img2, pred_landmarks)
            visualizer.landmarks_color = (0,0,255)
            # img = visualizer.draw_landmarks(img, landmarks)
            visualizer.show(img2, wait=False, winname="black")
            visualizer.show(img)
            print('*'*70,'\n')

            if visualizer.user_press == 27:
                break

def overfit_one_mini_batch():

    # ========= dataset ===========
    dataloader = create_test_loader(batch_size=20)
    # =========== models ============= 
    pfld_model = PFLD().to(device)
    auxiliary_model = AuxiliaryNet().to(device)

    pfld_model.train()
    auxiliary_model.train()
    criterion = PFLD_L2Loss().to(device)
    parameters = list(pfld_model.parameters()) + list(auxiliary_model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=0.0001, weight_decay=1e-6)
    
    image, labels = next(iter(dataloader))
    print(image.shape)
    time.sleep(5)
    for i in range(6000):
        euler_angles = labels['euler_angles'].squeeze() # shape (batch, 3)
        attributes = labels['attributes'].squeeze() # shape (batch, 6)
        landmarks = labels['landmarks'].squeeze() # shape (batch, 98, 2)
        landmarks = landmarks.reshape((landmarks.shape[0], 196)) # reshape landmarks to match loss function

        image = image.to(device)
        landmarks = landmarks.to(device)
        euler_angles = euler_angles.to(device)
        attributes = attributes.to(device)
        pfld_model = pfld_model.to(device)
        auxiliary_model = auxiliary_model.to(device)
 
        featrues, pred_landmarks = pfld_model(image)
        pred_angles = auxiliary_model(featrues)
        weighted_loss, loss = criterion(pred_landmarks, landmarks, pred_angles, euler_angles, attributes)

        train_w_loss = round(weighted_loss.item(),3)
        train_loss = round(loss.item(),3)
        print(f"\t.. weighted_loss= {train_w_loss} ... loss={train_loss}")

        optimizer.zero_grad()
        weighted_loss.backward()
        optimizer.step()

def show_model_tensorboard():
    import torch.utils.tensorboard as tensorboard

    pfld = PFLD()
    auxiliarynet = AuxiliaryNet()

    dataloader = create_test_loader(batch_size=20, transform=True)
    dataitr = iter(dataloader)
    image, labels = dataitr.next()
    print("ray2")
    writer = tensorboard.SummaryWriter("checkpoint/ray22")
    writer.add_graph(pfld, image)
    print("added model to tensorboard")
    time.sleep(4)
    writer.close()
    

if __name__ == "__main__":
    main()
    # overfit_one_mini_batch()
    # show_model_tensorboard()
"""
Author: Amr Elsersy
email: amrelsersay@gmail.com
-----------------------------------------------------------------------------------
Description: training script
"""
import numpy as np 
import argparse
import logging
import time

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
visualizer = WFLW_Visualizer()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10, help='num of training epochs')
    parser.add_argument('--batch_size', type=int, default=10, help="training batch size")
    parser.add_argument('--tensorboard', type=str, default='checkpoint/tensorboard', help='path log dir of tensorboard')
    parser.add_argument('--logging', type=str, default='checkpoint/logging', help='path of logging')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='optimizer weight decay')
    parser.add_argument('--datapath', type=str, default='data/WFLW', help='root path of WFLW dataset')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    writer = tensorboard.SummaryWriter(args.tensorboard)

    # dataloaders
    train_dataloader = create_train_loader(root=args.datapath,batch_size=args.batch_size, transform=True)
    test_dataloader  = create_test_loader(root=args.datapath, batch_size=1, transform=True)    
    # models & loss  
    pfld = PFLD().to(device)
    auxiliarynet = AuxiliaryNet().to(device)
    loss = PFLD_L2Loss().to(device)
    # optimizer & scheduler
    parameters = list(pfld.parameters()) + list(auxiliarynet.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        train_one_epoch(pfld, auxiliarynet, loss, optimizer, train_dataloader, epoch)
        validate(pfld, auxiliarynet, loss, test_dataloader, epoch)

def train_one_epoch(pfld_model, auxiliary_model, criterion, optimizer, dataloader, epoch_idx):

    for batch, (image, labels) in enumerate(dataloader):
        print(f"************************ batch {batch}/750  epoch {epoch_idx} ************************")

        image = image.to(device) # shape (batch, 3, 112, 112)
        landmarks = labels['landmarks'].squeeze() # shape (batch, 98, 2)
        euler_angles = labels['euler_angles'].squeeze() # shape (batch, 3)
        attributes = labels['attributes'].squeeze() # shape (batch, 6)
        rect = labels['rect'].squeeze()

        featrues, pred_landmarks = pfld_model(image)
        pred_angles = auxiliary_model(featrues)

        print("pred_landmarks",pred_landmarks.shape)
        print("pred_angles",pred_angles.shape)

        pred_landmarks = pred_landmarks.reshape((pred_landmarks.shape[0], 98, 2))

        weighted_loss, loss = criterion(pred_landmarks, landmarks, pred_angles, euler_angles, attributes)
        print("weighted_loss=",weighted_loss.item(), " ... loss=", loss.item())

        optimizer.zero_grad()
        weighted_loss.backward()
        optimizer.step()


def validate(pfld_model, auxiliary_model, criterion, dataloader, epoch_idx):
    with torch.no_grad:
        pass

        # # remove that after testing
        # l = {}
        # l['landmarks'] = landmarks[0].numpy()
        # l['euler_angles'] = euler_angles[0].numpy()
        # l['attributes'] = attributes[0].numpy()
        # l['rect'] = rect[0].numpy()
        # img = image[0]

        # print("image",image.shape)
        # print("landmarks",landmarks.shape)
        # print("euler",euler_angles.shape)
        # print("attributes",attributes.shape)

if __name__ == "__main__":
    main()
    # logging.log("Ray2")

    import torchvision.transforms.transforms as transforms
    transform = transforms.Compose([transforms.ToTensor()])    
    dataset = WFLW_Dataset(mode='train', transform=True)
    # # ============ From tensor to image ... crahses in any function in cv2 ==================
    # dataloader = create_train_loader(transform=True)
    # for images, labels in dataloader:
    #     print(images.shape)
    #     image = images[0]
    #     print(image.shape)
    #     image = to_numpy_image(images[0])
    #     print(image.shape)
    #     import cv2
    #     landmarks = labels['landmarks'].squeeze()[0]
    #     euler_angles = labels['euler_angles'].squeeze()[0]
    #     attributes = labels['attributes'].squeeze()[0]
    #     rect = labels['rect'].squeeze()[0]
    #     l = {}
    #     l['landmarks'] = landmarks.numpy()
    #     l['euler_angles'] = euler_angles.numpy()
    #     l['attributes'] = attributes.numpy()
    #     l['rect'] = rect.numpy()
    #     visualizer = WFLW_Visualizer()
    #     visualizer.visualize(image, l)

    #     print(image.shape, image)
    # ====================================================
    # ======= Habd
    # cv2.circle(image, (40,50), 30, (245,0,0), -1) 
    # cv2.imshow("I", image)
    # cv2.waitKey(0)  

    # datase2 = WFLW_Dataset(transform=False, mode='val')
    # image2, labels2 = datase2[0]
    # print(image.shape, image2.shape, type(image), type(image2))

    # ============= Test reshape and back reshape (works well) ======
    # x = np.array([[
    #     [1,2],
    #     [3,4],
    #     [5,6]
    # ]])
    # print(x.shape)
    # xx = transform(x)
    # print("transform",xx,'\n')
    # xx = xx.reshape((1,1,6))
    # print("flatten",xx,'\n')
    # xx = xx.reshape((1,3,2))
    # print("reshape",xx,'\n')
    
    # ======== Test Landmarks reshape =========
    # x = torch.tensor([
    #     1,2,3,4,5,6
    # ])
    # print(x.shape)
    # x = x.reshape((1,3,2))
    # print("reshape",x,'\n')

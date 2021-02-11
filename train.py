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
    parser.add_argument('--epochs', type=int, default=300, help='num of training epochs')
    parser.add_argument('--batch_size', type=int, default=20, help="training batch size")
    parser.add_argument('--tensorboard', type=str, default='checkpoint/tensorboard', help='path log dir of tensorboard')
    parser.add_argument('--logging', type=str, default='checkpoint/logging', help='path of logging')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='optimizer weight decay')
    parser.add_argument('--datapath', type=str, default='data/WFLW', help='root path of WFLW dataset')
    parser.add_argument('--pretrained', type=str,default='checkpoint/model_weights/weights.pth.tar',help='load checkpoint')
    parser.add_argument('--resume', action='store_true', help='resume from pretrained path specified in prev arg')
    parser.add_argument('--savepath', type=str, default='checkpoint/model_weights/weights.pth.tar', help='save checkpoint')    
    parser.add_argument('--savefreq', type=int, default=2, help="save weights each freq num of epochs")
    args = parser.parse_args()
    return args

# ======================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = parse_args()
writer = tensorboard.SummaryWriter(args.tensorboard)

def main():
    # ========= dataloaders ===========
    train_dataloader = create_train_loader(root=args.datapath,batch_size=args.batch_size, transform=True)
    test_dataloader  = create_test_loader(root=args.datapath, batch_size=args.batch_size, transform=True)    
    # ======== models & loss ========== 
    pfld = PFLD().to(device)
    auxiliarynet = AuxiliaryNet().to(device)
    loss = PFLD_L2Loss().to(device)
    # =========== optimizer =========== 
    parameters = list(pfld.parameters()) + list(auxiliarynet.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
    # ========= load weights ===========
    if args.resume:
        checkpoint = torch.load(args.pretrained)
        pfld.load_state_dict(checkpoint["pfld"])
        auxiliarynet.load_state_dict(checkpoint["auxiliary"])
        print(f'\tLoaded checkpoint from {args.pretrained}\n')
        time.sleep(1)
    else:
        print("******************* Start training from scratch *******************\n")
        time.sleep(5)
    # ========================================================================
    for epoch in range(args.epochs):
        # =========== train / validate ===========
        w_train_loss, train_loss = train_one_epoch(pfld, auxiliarynet, loss, optimizer, train_dataloader, epoch)
        val_loss = validate(pfld, auxiliarynet, loss, test_dataloader, epoch)
        # ============= tensorboard =============
        writer.add_scalar('train_weighted_loss',w_train_loss, epoch)
        writer.add_scalar('train_loss',train_loss, epoch)
        writer.add_scalar('val_loss',val_loss, epoch)
        # ============== save model =============
        if epoch % args.savefreq == 0:
            checkpoint_state = {
                "pfld": pfld.state_dict(),
                "auxiliary": auxiliarynet.state_dict()
            }
            torch.save(checkpoint_state, args.savepath)
            print(f'\n\t*** Saved checkpoint in {args.savepath} ***\n')
            time.sleep(2)
    writer.close()

def train_one_epoch(pfld_model, auxiliary_model, criterion, optimizer, dataloader, epoch):
    weighted_loss = 0
    loss = 0
    pfld_model.train()
    auxiliary_model.train()

    for image, labels in tqdm(dataloader):
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
        print(f"\ttraining epoch={epoch} .. weighted_loss= {train_w_loss} ... loss={train_loss}")

        optimizer.zero_grad()
        weighted_loss.backward()
        optimizer.step()

    return weighted_loss.item(), loss.item()    


def validate(pfld_model, auxiliary_model, criterion, dataloader, epoch):
    validation_losses = []
    pfld_model.eval()
    auxiliary_model.eval()

    with torch.no_grad():
        for image, labels in tqdm(dataloader):

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

            weighted_loss = round(weighted_loss.item(),3)
            loss = round(loss.item(),3)
            print(f"\tval epoch={epoch} .. val_weighted_loss= {weighted_loss} ... val_loss={loss}\n")
            
            validation_losses.append(loss)

        avg_val_loss = round(np.mean(validation_losses).item(),3)
               
        print('*'*70,f'\n\tEvaluation average loss= {avg_val_loss}\n')
        time.sleep(1)
        return avg_val_loss


if __name__ == "__main__":
    main()

    # import torchvision.transforms.transforms as transforms
    # transform = transforms.Compose([transforms.ToTensor()])    
    # dataset = WFLW_Dataset(mode='train', transform=True)
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

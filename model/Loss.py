import torch
import torch.nn as nn
import numpy as np 

"""
    Weighted L2 Loss Function that computes the  Sum(weight * |y` - y|^2)
    Sum is denoted for landmarks num
    then it is avaraged over the batch examples

    weight is function of euler angles & attributes of each example
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PFLD_L2Loss(nn.Module):
    def __init__(self):
        super(PFLD_L2Loss, self).__init__()
        self.l2_loss = nn.MSELoss(reduction='sum')
        

    def forward(self, landmarks, gt_landmarks, angles, gt_angles, attributes):
        

        # sum all 3 angles (by axis 1) for each batch example (mean on angles)
        # note torch.cos accepts only radians
        # batch_size = landmarks.shape[0]
        # angles_weight.shape (batch_size, 1)
        # print(angles, gt_angles)
        # angles_weight = torch.sum(1-torch.cos(torch.deg2rad(angles-gt_angles)), axis=1)

        # attributes weight .... v1
        # attributes_w_n = attributes[:, 1:6].float()
        # mat_ratio = torch.mean(attributes_w_n, axis=0)
        # mat_ratio = torch.Tensor([1.0 / (x) if x > 0 else batch_size for x in mat_ratio]).to(device)
        # attributes_weight = torch.sum(attributes_w_n.mul(mat_ratio), axis=1)

        
        to_radians = 0.0174532925
        diff = (angles-gt_angles)
        # it should be converted to radians .. but since diff is small the weight will be allways samll, so it is better to deal with degrees
        # diff *= to_radians 
        angles_weight = torch.sum(1-torch.cos(diff), axis=1)
        
        # attributes weight .... v2
        attributes = attributes.float()
        attributes_weight = torch.sum(attributes, axis=1)        

        # if we don't get the max .. all attributes =0 so weight will be 0 even if there is an error in
        # landmarks & angle, so we add a hing 1 to that weight to limit that .. same for angles
        attributes_weight += 1
        angles_weight += 1

        # L2 Landmarks Loss
        # shape (batch_size, 1) ... mean on both axes(1,2) to sum all x & all y seperatly them sum them
        landmarks_loss = torch.sum((landmarks-gt_landmarks)**2, 1)

        # print("landmakrs loss", landmarks_loss)
        # print(f"\nangles_weight: {angles_weight}") 
        # print(f"\nattributes_weight: {attributes_weight}")
        # mean on batch size
        return torch.mean(attributes_weight * angles_weight * landmarks_loss) , torch.mean(landmarks_loss)

    # wing_loss
    def wing_loss(self, y_true, y_pred, w=10.0, epsilon=2.0):
        y_pred = y_pred.reshape(-1, 98, 2)
        y_true = y_true.reshape(-1, 98, 2)

        x = y_true - y_pred
        c = w * (1.0 - math.log(1.0 + w / epsilon))
        absolute_x = torch.abs(x)
        losses = torch.where(w > absolute_x,
                            w * torch.log(1.0 + absolute_x / epsilon),
                            absolute_x - c)
        loss = torch.mean(torch.sum(losses, axis=[1, 2]), axis=0)
        return loss

if __name__ == "__main__":
    batch_size= 1
    landmarks = torch.randn((batch_size, 196)) * 255
    gt_landmarks = torch.randn((batch_size, 196)) * 255

    angles = torch.randn((batch_size, 3)) * 360
    gt_angles = torch.randn((batch_size, 3)) * 360
    # attributes = torch.randn((batch_size,6))
    attributes = torch.zeros((batch_size, 6))
    # attributes[:, 1] = 0
    # attributes[0:2, 3:5] = 0
    # attributes[:,3:5] = 1
    loss = PFLD_L2Loss()
    print(loss(landmarks, gt_landmarks, angles, gt_angles, attributes))

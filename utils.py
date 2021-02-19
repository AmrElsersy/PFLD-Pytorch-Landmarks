"""
Author: Amr Elsersy
email: amrelsersay@gmail.com
-----------------------------------------------------------------------------------
Description: utils functions for Data Augumentation & euler utils
"""
import cv2
import numpy as np
import torch
from euler_angles import EulerAngles
import math

def to_numpy_image(tensor):
    return np.transpose(tensor.numpy(), (1, 2, 0))

# =========== Data Augumentation ===================

def rotatedRectWithMaxArea(side, angle):
    """
    Given a square image of size side x side that has been rotated by 'angle' 
    (in degree), computes the new side of the largest possible
    axis-aligned square (maximal area) within the rotated image.
    """
    # convert to radians
    angle = angle * math.pi/180
    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))

    if side <= 2.*sin_a*cos_a*side or abs(sin_a-cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5*side
        new_side = x/sin_a,x/cos_a
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a*cos_a - sin_a*sin_a
        new_side = side*(cos_a -sin_a)/cos_2a

    return int(new_side)

def flip(image, landmarks):
    # horizontal flip
    image = cv2.flip(image, 1)

    w,h = image.shape[:2]
    center = (w//2, h//2)

    # translate it to origin
    landmarks -= center 
    # apply reflection(flip) matrix
    flip_matrix = np.array([
        [-1, 0],
        [0 , 1]
    ])   
    landmarks = (flip_matrix @ landmarks.T).T
    # translate again to its position
    landmarks += center

    # just flip the order of landmarks points .. mask is from https://github.com/polarisZhao/PFLD-pytorch/blob/master/data/Mirror98.txt
    flip_mask = [   32,31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12
                    ,11,10,9,8,7,6,5,4,3,2,1,0,46,45,44,43,42,50,49,48,47,37,36,35,
                    34,33,41,40,39,38,51,52,53,54,59,58,57,56,55,72,71,70,69,68,75,
                    74,73,64,63,62,61,60,67,66,65,82,81,80,79,78,77,76,87,86,85,84,
                    83,92,91,90,89,88,95,94,93,97,96]

    landmarks = landmarks[flip_mask]

    return image, landmarks

def resize(image, landmarks, size=(112,112)):
    side = image.shape[0]    
    scale = size[0] / side
    image = cv2.resize(image, size)
    landmarks *= scale
    return image, landmarks


# ============= Euler ==================
def euler_to_rotation(euler_angles) :
    R_x = np.array([[1,           0,                       0              ],
                    [0,  np.cos(np.radians(euler_angles[0])), -np.sin(np.radians(euler_angles[0]))],
                    [0,  np.sin(np.radians(euler_angles[0])),  np.cos(np.radians(euler_angles[0]))]
                    ])
                     
    R_y = np.array([[np.cos(np.radians(euler_angles[1])),    0,      np.sin(np.radians(euler_angles[1]))  ],
                    [0,                     1,   0                   ],
                    [-np.sin(np.radians(euler_angles[1])),   0,      np.cos(np.radians(euler_angles[1]))  ]
                    ])
                     
    R_z = np.array([[np.cos(np.radians(euler_angles[2])),    -np.sin(np.radians(euler_angles[2])),    0],
                    [np.sin(np.radians(euler_angles[2])),    np.cos(np.radians(euler_angles[2])),     0],
                    [0,                     0,                      1]
                    ])
                     
                     
    R = R_x @ R_y @ R_z
    return R        

def get_intrensic_matrix(image):
    e = EulerAngles((image.shape[0], image.shape[1]))
    return e.camera_intrensic_matrix


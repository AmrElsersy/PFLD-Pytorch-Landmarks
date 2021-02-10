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

def to_numpy_image(tensor):
    return np.transpose(tensor.numpy(), (1, 2, 0))

# =========== Data Augumentation ===================
def rotate(image, labels, angle):
    # rotate image
    # rotate landmarks
    # change euler (just yaw)
    return image, labels

def scale(image, labels):
    # fx, fy
    # scale image
    # scale landmarks with same scale
    return image, labels

def flip(image, labels):
    # flip image
    # flip landmarks
    # flip euler roll angle
    return image, labels

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


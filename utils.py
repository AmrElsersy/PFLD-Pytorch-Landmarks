"""
Author: Amr Elsersy
email: amrelsersay@gmail.com
-----------------------------------------------------------------------------------
Description: utils functions for Data Augumentation & euler utils
"""

import cv2
import numpy as np
from euler_angles import EulerAngles

# =========== Data Augumentation ===================
def rotate(image, labels, angle):
    # rotate image
    # rotate landmarks
    # change euler (just yaw)
    pass

def scale(image, labels):
    # fx, fy
    # scale image
    # scale landmarks with same scale
    pass

def flip(image, labels):
    # flip image
    # flip landmarks
    # flip euler roll
    pass

# ============= Euler ==================
def get_intrensic_matrix(image):
    e = EulerAngles((image.shape[0], image.shape[1]))
    return e.camera_intrensic_matrix

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


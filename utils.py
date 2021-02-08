import cv2
import numpy as np
from euler_angles import EulerAngles

# =========== Data Augumentation ===================
def rotate(image, labels):
    pass

def scale(image, labels):
    pass

def flip(image, labels):
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


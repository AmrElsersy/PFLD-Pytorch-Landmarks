"""
Author: Amr Elsersy
email: amrelsersay@gmail.com
-----------------------------------------------------------------------------------
Description: utils script for helper functions

# written with help of the official implemenation @ https://github.com/polarisZhao/PFLD-pytorch/blob/master/pfld/utils.py
"""

import cv2
import numpy as np

def ilimination(image):
    return image

def eular_angles_from_landmarks(landmarks_2D):
    """
        Pitch, Yaw, Roll Rotation angles (eular angles) from 2D-3D Correspondences Landmarks
        Givin a General 3D Face Model (3D Landmarks) & annotations 2D landmarks
        2D point = internsic * exterinsic * 3D point_in_world_space
        if we have 2D - 3D correspondences & internsic camera matrix, 
        we can use cv2.solvPnP to get the extrensic matrix that convert the world_space to camera_3D_space
        this extrensic matrix is considered as the 3 eular angles

        we can do that because the faces have similar 3D structure and emotions & iluminations dosn't affect the pose
        Notes:
            we can choose get any 3D coord from any 3D face model .. changing translation will not affect the angle
            it will only inroduce a bigger tvec but the same rotation matrix
    """

    # WFLW(98 landmark) tracked points
    TRACKED_POINTS_MASK = [33, 38, 50, 46, 60, 64, 68, 72, 55, 59, 76, 82, 85, 16]
    landmarks_2D = landmarks[TRACKED_POINTS_MASK]

    
    # X-Y-Z with X pointing forward and Y on the left and Z up (same as LIDAR)
    # OpenCV Coord X points to the right, Y down, Z to the front
    
    # General 3D Face Model Coordinates (3D Landmarks)
    landmarks_3D = np.float32([
        [6.825897, 6.760612, 4.402142],  # LEFT_EYEBROW_LEFT, 
        [1.330353, 7.122144, 6.903745],  # LEFT_EYEBROW_RIGHT, 
        [-1.330353, 7.122144, 6.903745],  # RIGHT_EYEBROW_LEFT,
        [-6.825897, 6.760612, 4.402142],  # RIGHT_EYEBROW_RIGHT,
        [5.311432, 5.485328, 3.987654],  # LEFT_EYE_LEFT,
        [1.789930, 5.393625, 4.413414],  # LEFT_EYE_RIGHT,
        [-1.789930, 5.393625, 4.413414],  # RIGHT_EYE_LEFT,
        [-5.311432, 5.485328, 3.987654],  # RIGHT_EYE_RIGHT,
        [-2.005628, 1.409845, 6.165652],  # NOSE_LEFT,
        [-2.005628, 1.409845, 6.165652],  # NOSE_RIGHT,
        [2.774015, -2.080775, 5.048531],  # MOUTH_LEFT,
        [-2.774015, -2.080775, 5.048531],  # MOUTH_RIGHT,
        [0.000000, -3.116408, 6.097667],  # LOWER_LIP,
        [0.000000, -7.415691, 4.070434],  # CHIN
    ])

    # Camera internsic Matrix
    c_x = 128
    f_x = c_x / np.tan(np.radians(30))
    c_y = c_x
    f_y = f_x

    camera_internsic_matrix = np.float32([
        [f_x, 0.0, c_x], 
        [0.0, f_y, c_y],
        [0.0, 0.0, 1.0]
        ])


    """
        solve for extrensic matrix (rotation & translation) with 2D-3D correspondences
        returns:
            rvec: rotation vector (as rotation is 3 degree of freedom, it is represented as 3d-vector)
            tvec: translate vector (world origin position relative to the camera 3d coord system)
            _ : error -not important-.
    """
    _, rvec, tvec = cv2.solvePnP(landmarks_3D, landmarks_2D, camera_internsic_matrix, distCoeffs=None)
    # convert rotation vector to rotation matrix .. note: function is used for vice versa
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    # [R T]
    extrensic_matrix = np.hstack((rotation_matrix, tvec))    
    # decompose the extrensic matrix to many things including the 3 eular angles
    euler_angles = cv2.decomposeProjectionMatrix(extrensic_matrix)[6]

    return {
        'pitch': euler_angles[0],
        'roll' : euler_angles[1],
        'yaw'  : euler_angles[2]
    }


if __name__ == "__main__":
    from dataset import WFLW_Dataset
    from visualization import WFLW_Visualizer

    dataset = WFLW_Dataset()
    visualizer = WFLW_Visualizer()

    for i in range(len(dataset)):
        image, labels = dataset[i]
        landmarks = labels['landmarks']
        eular_angles_from_landmarks(landmarks)
        
        visualizer.visualize(image, labels)




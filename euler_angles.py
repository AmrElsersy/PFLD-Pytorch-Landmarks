"""
Author: Amr Elsersy
email: amrelsersay@gmail.com
-----------------------------------------------------------------------------------
Description: Head Pose Euler Angles(pitch yaw roll) estimation from 2D-3D correspondences landmarks
"""

import cv2
import numpy as np

class EulerAngles:
    """
        Head Pose Estimation from landmarks annotations with solvePnP OpenCV
        Pitch, Yaw, Roll Rotation angles (eular angles) from 2D-3D Correspondences Landmarks

        Givin a General 3D Face Model (3D Landmarks) & annotations 2D landmarks
        2D point = internsic * exterinsic * 3D point_in_world_space
        if we have 2D - 3D correspondences & internsic camera matrix, 
        we can use cv2.solvPnP to get the extrensic matrix that convert the world_space to camera_3D_space
        this extrensic matrix is considered as the 3 eular angles & translation vector

        we can do that because the faces have similar 3D structure and emotions & iluminations dosn't affect the pose
        Notes:
            we can choose get any 3D coord from any 3D face model .. changing translation will not affect the angle
            it will only inroduce a bigger tvec but the same rotation matrix
    """

    def __init__(self,  img_shape=(112,112) ):
        # Lazy Estimation of Camera internsic Matrix Approximation
        self.camera_intrensic_matrix = self.estimate_camera_matrix(img_shape)        
        # 3D Face model 3D landmarks
        self.landmarks_3D = self.get_face_model_3D_landmarks()

    def estimate_camera_matrix(self, img_shape):
        # Used Weak Prespective projection as we assume near object with similar depths

        # cx, cy the optical centres
        # translation to image center as image center here is top left corner
        # focal length is function of image size & Field of View (assumed to be 30 degree)
        c_x = img_shape[0] / 2
        c_y = img_shape[1] / 2
        FieldOfView = 60
        focal = c_x / np.tan(np.radians(FieldOfView/2))
        
        # Approximated Camera intrensic matrix assuming weak prespective
        return np.float32([
            [focal, 0.0,    c_x], 
            [0.0,   focal,  c_y],
            [0.0,   0.0,    1.0]
        ])

    def set_img_shape(self, img_shape):
        self.camera_intrensic_matrix = self.estimate_camera_matrix(img_shape)

    def get_face_model_3D_landmarks(self):
        """
            General 3D Face Model Coordinates (3D Landmarks) 
            obtained from antrophometric measurement of the human head.

            Returns:
            -------
            3D_Landmarks: numpy array of shape(N, 3) as N = 11 point in 3D
        """
        # X-Y-Z with X pointing forward and Y on the left and Z up (same as LIDAR)
        # OpenCV Coord X points to the right, Y down, Z to the front (same as 3D Camera)
        
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

        return landmarks_3D


    def eular_angles_from_landmarks(self, landmarks_2D):
        """
            Estimates Euler angles from 2D landmarks 
            
            Parameters:
            ----------
            landmarks_2D: numpy array of shape(N, 2) as N is num of landmarks (usualy 98 from WFLW)

            Returns:
            -------
            rvec: rotation numpy array that transform model space to camera space (3D in both)
            tvec: translation numpy array that transform model space to camera space
            euler_angles: (pitch yaw roll) in degrees
        """

        # WFLW(98 landmark) tracked points
        TRACKED_POINTS_MASK = [33, 38, 50, 46, 60, 64, 68, 72, 55, 59, 76, 82, 85, 16]
        landmarks_2D = landmarks_2D[TRACKED_POINTS_MASK]

        """
            solve for extrensic matrix (rotation & translation) with 2D-3D correspondences
            returns:
                rvec: rotation vector (as rotation is 3 degree of freedom, it is represented as 3d-vector)
                tvec: translate vector (world origin position relative to the camera 3d coord system)
                _ : error -not important-.
        """
        _, rvec, tvec = cv2.solvePnP(self.landmarks_3D, landmarks_2D, self.camera_intrensic_matrix, distCoeffs=None)

        """
            note:
                tvec is almost constant = the world origin coord with respect to the camera
                avarage value of tvec = [-1,-2,-21]
                we can use this directly without computing tvec
        """

        # convert rotation vector to rotation matrix .. note: function is used for vice versa
        # rotation matrix that transform from model coord(object model 3D space) to the camera 3D coord space
        rotation_matrix, _ = cv2.Rodrigues(rvec)

        # [R T] may be used in cv2.decomposeProjectionMatrix(extrinsic)[6]
        extrensic_matrix = np.hstack((rotation_matrix, tvec))    
        
        # decompose the extrensic matrix to many things including the 3 eular angles 
        # (pitch yaw roll) in degrees
        euler_angles = cv2.RQDecomp3x3(rotation_matrix)[0]

        return rvec, tvec, euler_angles

if __name__ == "__main__":
    from dataset import WFLW_Dataset
    from visualization import WFLW_Visualizer

    dataset = WFLW_Dataset(mode='train')
    visualizer = WFLW_Visualizer()
    eular_estimator = EulerAngles()        

    for i in range(len(dataset)):
        image, labels = dataset[i]
        landmarks = labels['landmarks']

        rvec, tvec, euler_angles = eular_estimator.eular_angles_from_landmarks(landmarks)
        image = visualizer.draw_euler_angles(image, rvec, tvec, euler_angles, eular_estimator.camera_intrensic_matrix)

        print("rvec\n", rvec)
        print("tvec\n", tvec)
        print("euler ", euler_angles)        
        print ("*" * 80, '\n\n\t press n for next example .... ESC to exit')
        visualizer.show(image)

        if visualizer.user_press == 27:
            cv2.destroyAllWindows()
            break
        



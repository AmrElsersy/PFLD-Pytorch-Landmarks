import sys
import time
import argparse
import cv2
import numpy as np
from numpy.lib.type_check import imag
import torch
import torchvision.transforms.transforms as transforms
from face_detector.face_detector import DnnDetector, HaarCascadeDetector
from model.model import PFLD
from euler_angles import EulerAngles

sys.path.insert(1, 'face_detector')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def draw_euler_angles(frame, face, axis_pts, euler_angles):
    (x,y,w,h) = face
    top_left = (x,y)
    center = (x+w//2, y+h//2)

    axis_pts = axis_pts.astype(np.int32)
    pitch_point = tuple(axis_pts[0].ravel() + top_left)
    yaw_point   = tuple(axis_pts[1].ravel() + top_left)
    roll_point  = tuple(axis_pts[2].ravel() + top_left)

    width = 3
    cv2.line(frame, center,  pitch_point, (0,255,0), width)
    cv2.line(frame, center,  yaw_point, (255,0,0), width)
    cv2.line(frame, center,  roll_point, (0,0,255), width)

    pitch, yaw, roll = euler_angles
    cv2.putText(frame, "Pitch:{:.2f}".format(pitch), (x,y+10), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0))
    cv2.putText(frame, "Yaw:{:.2f}".format(yaw), (x,y+25), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0))
    cv2.putText(frame, "Roll:{:.2f}".format(roll), (x,y+40), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0))

    return frame

def main(args):
    # Models
    pfld = PFLD().to(device)
    pfld.eval()
    head_pose = EulerAngles()

    # Load model
    checkpoint = torch.load(args.pretrained, map_location=device)
    pfld.load_state_dict(checkpoint['pfld'])

    # Face detection
    root = 'face_detector'
    face_detector = DnnDetector(root)
    if args.haar:
        face_detector = HaarCascadeDetector(root)

    video = cv2.VideoCapture(0) # 480, 640
    # video = cv2.VideoCapture("4.mp4") # (720, 1280) or (1080, 1920)
    t1 = 0
    t2 = 0
    while video.isOpened():
        _, frame = video.read()

        # time
        t2 = time.time()
        fps = round(1/(t2-t1))
        t1 = t2

        # faces
        faces = face_detector.detect_faces(frame)

        for face in faces:
            (x,y,w,h) = face
            side = max(w,h)
            d_side = int(0.1 * side)
            x -= d_side//2
            y -= d_side//2
            w += d_side
            h += d_side

            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 3)

            # preprocessing
            t = time.time()
            input_face = frame[y:y+h, x:x+w]
            input_face = cv2.resize(input_face, (112,112))
            input_face = transforms.ToTensor()(input_face).to(device)
            input_face = torch.unsqueeze(input_face, 0)

            with torch.no_grad():
                # landmarks
                _, landmarks = pfld(input_face)
                # print(f'PFLD Forward time = {(time.time()-t)*1000}')
                # visualization
                landmarks = landmarks.cpu().reshape(98,2).numpy()
                landmarks = (landmarks * (w,h) ).astype(np.int32) 
    
                for (x_l, y_l) in landmarks:
                    cv2.circle(frame, (x + x_l, y + y_l), 2, (0,255,0), -1)

                if args.head_pose:
                    rvec, tvec, euler_angles = head_pose.eular_angles_from_landmarks(np.copy(landmarks).astype(np.float))
                    axis = np.identity(3) * 7
                    axis_pts = cv2.projectPoints(axis, rvec, tvec, head_pose.camera_intrensic_matrix, None)[0]
                    frame = draw_euler_angles(frame, (x,y,w,h), axis_pts, euler_angles)

        cv2.putText(frame, str(fps), (10,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))
        cv2.imshow("Video", frame)   
        # cv2.imshow("black", face_frame)
        if cv2.waitKey(1) & 0xff == 27:
            video.release()
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--haar', action='store_true', help='run the haar cascade face detector')
    parser.add_argument('--pretrained',type=str,default='checkpoint/model_weights/weights.pth.tar',help='load weights')
    parser.add_argument('--head_pose', action='store_true', help='visualization of head pose euler angles')
    args = parser.parse_args()

    main(args)


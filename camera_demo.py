import sys
import time
import argparse
import cv2
import numpy as np
import torch
import torchvision.transforms.transforms as transforms
from face_detector.face_detector import DnnDetector, HaarCascadeDetector
from model.model import PFLD
from euler_angles import EulerAngles

sys.path.insert(1, 'face_detector')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        cv2.putText(frame, str(fps), (10,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))
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
    args = parser.parse_args()

    main(args)


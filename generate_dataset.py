"""
Author: Amr Elsersy
email: amrelsersay@gmail.com
-----------------------------------------------------------------------------------
Description: Dataset Augumentation & Generation
"""

import os, time, enum
from PIL import Image
import argparse
from numpy.lib.type_check import imag
import math
import numpy as np 
import cv2
from euler_angles import EulerAngles

# ============= Data Augumentation =============
from utils import rotate, flip, resize, rotatedRectWithMaxArea

class Data_Augumentor:
    """
        Data Augumentation
        - reads original dataset annotations and preprocess data & augument it
        - generates new & clean dataset ready to be used.
    """
    def __init__(self):
        self.face_shape = (112,112)
        self.theta1 = 15
        self.theta2 = 30
        self.euler_estimator = EulerAngles()

        self.root = 'data'
        self.train_path = os.path.join(self.root, 'train')
        self.test_path = os.path.join(self.root, 'test')

        self.images_root = os.path.join(self.root, 'WFLW', "WFLW_images")
        train_test_root = os.path.join(self.root, 'WFLW', "WFLW_annotations", "list_98pt_rect_attr_train_test")
        train_name = os.path.join(train_test_root, "list_98pt_rect_attr_train.txt")
        test_name = os.path.join(train_test_root, "list_98pt_rect_attr_test.txt")
        test_file  = open(test_name ,'r')
        train_file = open(train_name,'r')
        # the important ones
        self.test_lines  = test_file.read().splitlines()
        self.train_lines = train_file.read().splitlines()

    def scale_rect(self, rect, factor):
        (x1, y1), (x2, y2) = rect            
        rect_dw = (x2 - x1) * factor
        rect_dy = (y2 - y1) * factor
        x1 -= rect_dw/2
        x2 += rect_dw/2
        y1 -= rect_dy/2
        y2 += rect_dy/2
        x1 = max(x1, 0)
        x2 = max(x2, 0)
        y1 = max(y1, 0)
        y2 = max(y2, 0)
        rect[0] = (x1,y1)
        rect[1] = (x2,y2)
        return rect

    def generate_dataset(self, mode='train'):
        assert mode in ['train', 'test']
        try:
            if mode == 'train':
                os.mkdir(self.train_path)
                os.mkdir(os.path.join(self.train_path, 'images'))
            else:
                os.mkdir(self.test_path)
                os.mkdir(os.path.join(self.test_path, 'images'))           
            print(f'created data/{mode} folder')
        except:
            print(f"data/{mode} folder already exist .. delete it to generate a new dataset")
            return

        lines = self.train_lines if mode == 'train' else self.test_lines
        save_path = self.train_path if mode == 'train' else self.test_path

        # annotation for all train/test dataset strings
        all_annotations_str = []

        k = 0
        for annotations_line in lines:
            # read annotations
            annotations = self.read_annotations(annotations_line)
            print('k=',k)
            image_full_path = annotations['path']
            image = self.read_image(image_full_path)
            rect = annotations['rect']
            landmarks = annotations['landmarks']
            attributes = annotations['attributes']
                        
            # ============= Data Augumentation =================
            """
                8x dataset
                    original image + flip image
                    4 rotated image
                    2 rotated flip image
            """
            all_images = []
            all_landmarks = []

            if mode == 'test':
                image, rect, landmarks = self.crop_face(np.copy(image), np.copy(rect), np.copy(landmarks))
                all_images = [image]
                all_landmarks = [landmarks]
            else:
                original_image, _, original_landmarks = self.crop_face(np.copy(image), np.copy(rect), np.copy(landmarks))
                flip_original_image, flip_original_landmarks = flip(np.copy(original_image), np.copy(original_landmarks))

                all_images.append(original_image)
                all_landmarks.append(original_landmarks)
                all_images.append(flip_original_image)
                all_landmarks.append(flip_original_landmarks)

                # crop face & resize to a bigger rect
                scaled_rect = self.scale_rect(rect, 0.25)
                image, rect, landmarks = self.crop_face(np.copy(image), scaled_rect, np.copy(landmarks))
                flip_image, flip_landmarks = flip(np.copy(image), np.copy(landmarks))
                
                augumentation_angles = [30, -30, 15, -15]
                for angle in augumentation_angles:
                    rotated_image, rotated_landmarks = rotate(image, landmarks, angle)
                    all_images.append(rotated_image)
                    all_landmarks.append(rotated_landmarks)

                augumentation_angles_flip = [20, -20]
                for angle in augumentation_angles_flip:
                    rotated_image, rotated_landmarks = rotate(np.copy(flip_image), np.copy(flip_landmarks), angle)
                    all_images.append(rotated_image)
                    all_landmarks.append(rotated_landmarks)

            # for every augumented image
            for i, img in enumerate(all_images):
                img = all_images[i]
                landmark = all_landmarks[i]

                # # visualize
                # for point in landmark:
                #     point = (int(point[0]), int(point[1]))
                #     cv2.circle(img, point, 1, (0,255,0), -1)
                # img = cv2.resize(image, (300,300))
                # cv2.imshow("image", img)
                # if cv2.waitKey(0) == 27:
                #     exit(0)
                # print(image_full_path)
                # print("*"*80)

                # generate euler angles from landmarks
                _, _, euler_angles = self.euler_estimator.eular_angles_from_landmarks(landmark)
                euler_str = ' '.join([str(round(angle,2)) for angle in euler_angles])

                # get image name
                new_image_path = self.save_image(img, image_full_path, k, save_path) # id should be unique for every img

                # convert landmarks to string
                landmarks_list = landmark.reshape(196,).tolist()
                landmarks_str = ' '.join([str(l) for l in landmarks_list])

                # attributes list to string
                attributes_str = ' '.join([str(attribute) for attribute in attributes])

                # annotation string = image_name + 98 landmarks + attributes + euler
                new_annotation = ' '.join([new_image_path, landmarks_str, attributes_str, euler_str])
                all_annotations_str.append(new_annotation)
                # print(new_annotation)

            k += 1
            if k % 100 == 0:
                print(f'{mode} dataset: {k} generated data')

        # ========= Save annotations ===============
        one_annotation_str = '\n'.join([annotation for annotation in all_annotations_str])
        annotations_path = os.path.join(save_path, 'annotations.txt')
        annotations_file = open(annotations_path, 'w')
        annotations_file.write(one_annotation_str)
        annotations_file.close()
        print('*'*60,f'\n\t {mode} annotations is saved @ data/{mode}/annotations.txt')
        time.sleep(2)

    def crop_face(self, image, rect, landmarks):
        (x1, y1), (x2, y2) = rect           
        image = image[int(y1):int(y2), int(x1):int(x2)]

        # resize the image & store the dims to resize landmarks
        h, w = image.shape[:2]
        image = cv2.resize(image, self.face_shape)

        # scale factor in x & y to scale the landmarks
        new_h, new_w = self.face_shape
        fx = new_w / w
        fy = new_h / h
        # translate the landmarks then scale them
        landmarks -= rect[0]
        for landmark in landmarks:
            landmark[0] *= fx
            landmark[1] *= fy

        # face rect
        rect[0] = (0,0)
        rect[1] = (x2-x1, y2-y1)

        return image, rect, landmarks

    def save_image(self, img, full_name, id, save_path):
        full_name = full_name.split('/')
        image_name = full_name[-1][:-4] + '_' + str(id) + '.jpg'
        image_path = os.path.join(save_path, 'images', image_name)
        cv2.imwrite(image_path, img)
        return image_path

    def read_annotations(self, annotations):
        annotations = annotations.split(' ')

        landmarks = annotations[0:196]
        rect = annotations[196:200]        
        attributes = annotations[200:206]
        image_path = annotations[206]

        landmarks = [float(landmark) for landmark in landmarks]        
        landmarks = np.array(landmarks, dtype=np.float).reshape(98,2)
        rect = [float(coord) for coord in rect]
        rect = np.array(rect, dtype=np.float).reshape((2,2))

        return {
            'landmarks': landmarks,
            'rect' : rect,
            'attributes': attributes,
            'path': image_path
        }

    def read_image(self, name):
        path = os.path.join(self.images_root, name)
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        return image        



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train')
    args = parser.parse_args()

    augumentor = Data_Augumentor()
    augumentor.generate_dataset(args.mode)
    
if __name__ == "__main__":
    main()

"""
Author: Amr Elsersy
email: amrelsersay@gmail.com
-----------------------------------------------------------------------------------
Description: Dataset Augumentation & Generation
"""

import os, time
import argparse
from numpy.lib.type_check import imag
import math
import numpy as np 
import cv2
from euler_angles import EulerAngles

# ============= Data Augumentation =============
from utils import flip, resize

class Data_Augumentor:
    """
        Data Augumentation
        - reads dataset annotations and preprocess data & augument it
        - generates new & clean dataset ready to be used.
    """
    def __init__(self, n_augumentation=5):
        self.n_augumentation = n_augumentation
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
            image_full_path = annotations['path']
            image = self.read_image(image_full_path)
            rect = annotations['rect']
            landmarks = annotations['landmarks']
            attributes = annotations['attributes']

            # ============= Data Augumentation =================
            all_images = []
            all_landmarks = []

            if mode == 'test':
                image, landmarks, skip = self.crop_face_landmarks(image, landmarks, False)
                if skip:
                    continue
                all_images = [image]
                all_landmarks = [landmarks]
            else:
                for i in range(self.n_augumentation):
                    angle = np.random.randint(-30, 30)

                    augument_image, augument_landmarks = self.rotate(np.copy(image), np.copy(landmarks), angle)
                    augument_image, augument_landmarks, skip = self.crop_face_landmarks(augument_image, augument_landmarks)
                    if skip:
                        continue
                    
                    if np.random.choice((True, False)):
                        augument_image, augument_landmarks = flip(augument_image, augument_landmarks)

                    # # visualize
                    # img = np.copy(augument_image)
                    # for point in augument_landmarks:
                    #     point = (int(point[0]), int(point[1]))
                    #     cv2.circle(img, point, 1, (0,255,0), -1)
                    # # img = cv2.resize(img, (300,300))
                    # cv2.imshow("image", img)
                    # if cv2.waitKey(0) == 27:
                    #     exit(0)
                    # print("*"*80)

                    all_images.append(augument_image)
                    all_landmarks.append(augument_landmarks)

            # for every augumented image
            for i, img in enumerate(all_images):
                img = all_images[i]
                landmark = all_landmarks[i] / 112

                # generate euler angles from landmarks
                _, _, euler_angles = self.euler_estimator.eular_angles_from_landmarks(landmark)
                euler_str = ' '.join([str(round(angle,2)) for angle in euler_angles])

                # get image name
                new_image_path = self.save_image(img, image_full_path, k, i, save_path) # id should be unique for every img

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

    def rotate(self, image, landmarks, theta):
        top_left = np.min(landmarks, axis=0).astype(np.int32) 
        bottom_right = np.max(landmarks, axis=0).astype(np.int32)
        wh = bottom_right - top_left + 1
        center = (top_left + wh/2).astype(np.int32)
        boxsize = int(np.max(wh)*1.2)
        cx, cy = center

        # random shift
        cx += int(np.random.randint(-boxsize*0.1, boxsize*0.1))
        cy += int(np.random.randint(-boxsize*0.1, boxsize*0.1))

        center = (cx, cy)

        # get translation-rotation matrix numpy array shape (2,3) has rotation and last column is translation
        # note that it translate the coord to the origin apply the rotation then translate it again to cente
        rotation_matrix = cv2.getRotationMatrix2D(center, theta, 1)
        # to keep all the boxes is visible as some boundary boxes may dissapear during rotation
        shape_factor = 1.1
        h, w = image.shape[:2]
        new_shape = (int(w*shape_factor), int(h*shape_factor))
        image = cv2.warpAffine(image, rotation_matrix, new_shape)

        # add homoginous 1 to 2D landmarks to be able to use the same translation-rotation matrix
        landmarks =np.hstack((landmarks, np.ones((98, 1))))
        landmarks = (rotation_matrix @ landmarks.T).T

        # for point in landmarks:
        #     point = (int(point[0]), int(point[1]))
        #     cv2.circle(image, point, 0, (0,0,255), -1)
        # ima = cv2.resize(image, (500,500))
        # cv2.imshow("image", ima)
        # if cv2.waitKey(0) == 27:
        #     exit(0)

        return image, landmarks

    def crop_face_landmarks(self, image, landmarks, is_scaled=True):
        # max (x,y) together & the min is the boundary of bbox
        top_left = np.min(landmarks, axis=0).astype(np.int32) 
        bottom_right = np.max(landmarks, axis=0).astype(np.int32)
        
        x1,y1 = top_left
        x2,y2 = bottom_right
        rect = [(x1, y1), (x2, y2)]
 
        if is_scaled:
            wh = np.ptp(landmarks, axis=0).astype(np.int32) + 1
            scaled_size = np.random.randint(int(np.min(wh)), np.ceil(np.max(wh) * 1.25))            

            (x1, y1), (x2, y2) = self.scale_rect(rect, scaled_size, image.shape)
        else:
            (x1, y1), (x2, y2) = self.scale_rect2(rect, 0.2, image.shape)


        if x1 == x2 or y1 == y2:
            return None, None, True

        # landmarks normalization
        landmarks -= (x1,y1)
        landmarks = landmarks / [x2-x1, y2-y1] 

        # when rotation is applied, boundary parts of image may disapear & landmarks will be out of the image shape
        if (landmarks < 0).any() or (landmarks > 1).any() :
            return None, None, True

        # crop
        image = image[int(y1):int(y2), int(x1):int(x2)]
        # resize
        image = cv2.resize(image, self.face_shape)
        landmarks *= 112

        # skip this image if any of top left coord has a big -ve
        # because this will lead to a big shift to landmarks & wrong annotations
        skip = False
        min_neg = min(x1,y1)
        if min_neg < -5:
            skip = True

        return image, landmarks, skip

    def scale_rect(self, rect, factor, big_img_shape):
        (x1, y1), (x2, y2) = rect        
        cx = (x1+x2) // 2
        cy = (y1+y2) // 2    

        top_left = np.asarray((cx - factor // 2, cy - factor//2), dtype=np.int32)
        bottom_right = top_left + (factor, factor)

        (x1,y1) = top_left
        (x2,y2) = bottom_right

        x1 = max(x1, 0)
        y1 = max(y1, 0)
        h_max, w_max = big_img_shape[:2]
        y2 = min(y2, h_max)
        x2 = min(x2, w_max)
        rect[0] = (x1,y1)
        rect[1] = (x2,y2)
        return np.array(rect).astype(np.int32)
 
    def scale_rect2(self, rect, factor, big_img_shape):
        (x1, y1), (x2, y2) = rect            
        rect_dw = (x2 - x1) * factor
        rect_dy = (y2 - y1) * factor
        x1 -= rect_dw/2
        x2 += rect_dw/2
        y1 -= rect_dy/2
        y2 += rect_dy/2
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        h_max, w_max = big_img_shape[:2]
        y2 = min(y2, h_max)
        x2 = min(x2, w_max)
        rect[0] = (x1,y1)
        rect[1] = (x2,y2)
        return np.array(rect).astype(np.int32)

    def crop_face_rect(self, image, rect, landmarks):
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

    def save_image(self, img, full_name, k, id, save_path):
        full_name = full_name.split('/')
        image_name = full_name[-1][:-4] + '_' + str(k) + '_' + str(id) + '.jpg'
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
    parser.add_argument('--n', type=int, default=5, help='number of augumented images per image')
    args = parser.parse_args()

    augumentor = Data_Augumentor(args.n)
    augumentor.generate_dataset(args.mode)
    
if __name__ == "__main__":
    main()

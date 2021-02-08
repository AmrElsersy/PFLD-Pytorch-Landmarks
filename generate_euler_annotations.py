"""
Author: Amr Elsersy
email: amrelsersay@gmail.com
-----------------------------------------------------------------------------------
Description: Generate Euler angles annotations from annotated landmarks 
             Save them in the same annotations files at the end of each line
"""

from dataset import WFLW_Dataset
from euler_angles import EulerAngles
import time

def generate_euler_annotations(dataset, lines, file_path):
    print("\n\tStart generating euler annotations @ ", file_path)
    time.sleep(2)

    euler_estimator = EulerAngles()

    for i in range(len(dataset)):
        image, labels = dataset[i]
        
        # Euler angles
        landmarks = labels['landmarks']
        rvec, tvec, euler_angles = euler_estimator.eular_angles_from_landmarks(landmarks)        

        # append euler string to each line
        euler_str = ' '.join([str(round(angle,2)) for angle in euler_angles])
        lines[i] += ' ' + euler_str + '\n'
        print(f"{i}/{len(dataset)}\teuler angles added:", euler_str)

    file = open(file_path, 'w')
    file.writelines(lines)
    print("Success euler annotations saved @ ", file_path)
    time.sleep(2)
    file.close()

if __name__ == '__main__':

    train_dataset = WFLW_Dataset(mode='train')  
    test_dataset = WFLW_Dataset(mode='val')

    train_path = train_dataset.train_name
    test_path = test_dataset.test_name

    train_lines = train_dataset.train_lines
    test_lines = test_dataset.test_lines

    N_annotations_with_euler = 210

    if len(train_lines[0].split(' ')) != N_annotations_with_euler:
        generate_euler_annotations(train_dataset, train_lines, train_path)
    else:
        print("\n\tAlready annotated with euler angles @", train_path)

    if len(test_lines[0].split(' ')) != N_annotations_with_euler:
        generate_euler_annotations(test_dataset, test_lines, test_path)
    else:
        print("\n\tAlready annotated with euler angles @", train_path)


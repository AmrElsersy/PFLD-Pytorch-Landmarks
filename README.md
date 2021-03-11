
# Practical Facial Landmarks Detector 

My unofficial implementation of [PFLD paper](https://arxiv.org/pdf/1902.10859.pdf) "Practical Facial Landmarks Detector" using Pytorch for a real time landmarks detection and head pose estimation.<br/>

![pfld](https://user-images.githubusercontent.com/35613645/109653302-89e1ef00-7b69-11eb-8dd7-e8810deebe44.png)


#### Demo

<img src="https://user-images.githubusercontent.com/35613645/110829589-f8792800-82a0-11eb-833d-0d665503a869.gif" width="1100" height="300">


##### How to Install
```
 $ pip3 install -r requirements.txt
 
 # Note that it can be run on lower versions of Pytorch so replace the versions with yours
```

##### install opencv & dnn from source (optional)
Both opencv dnn & haar cascade are used for face detection, if you want to use haar cascade you can skip this part.
```
sudo apt update && sudo apt install -y cmake g++ wget unzip
wget -O opencv.zip https://github.com/opencv/opencv/archive/master.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/master.zip
unzip opencv.zip
unzip opencv_contrib.zip
mkdir -p build && cd build
cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-master/modules ../opencv-master
cmake --build .
```
if you have any problems, refere to [Install opencv with dnn from source](https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html)

##### Run Camera Demo
Live camera demo 
```python
$ python3 camera_demo.py

# add '--head_pose' option to visualize head pose directions 
$ python3 camera_demo.py --head_pose

# add '--haar' option if you want to use Haar cascade detector instead of dnn opencv face detector
$ python3 camera_demo.py --haar
```

##### WFLW Dataset
[Wider Facial Landmarks in-the-wild (WFLW)](https://wywu.github.io/projects/LAB/WFLW.html) contains 10000 faces (7500 for training and 2500 for testing) with 98 fully manual annotated landmarks.

**Download** the dataset & place it in '**data/WFLW**' folder path
- WFLW Training and Testing Images [Google Drive](https://drive.google.com/file/d/1hzBd48JIdWTJSsATBEB_eFVvPL1bx6UC/view)<br/>
- WFLW Face Annotations [Download](https://wywu.github.io/projects/LAB/support/WFLW_annotations.tar.gz)<br/>
##### Prepare the dataset
Dataset augumentation & preparation <br/>
(Only apply one of the 2 options) for data augumentation
```python
$ python3 generate_dataset.py
```
```python
# another option to augument dataset from polarisZhao/PFLD-pytorch repo 
$ cd data
$ python3 SetPreparation.py
```


##### Visualize dataset
Visualize dataset examples with annotated landmarks & head pose 
```cmd
# add '--mode' option to determine the dataset to visualize
$ python3 visualization.py
```
##### Visualize euler angles
```cmd
$ python3 euler_angles.py
```

##### Tensorboard 
Take a wide look on dataset examples using tensorboard
```
$ python3 visualization.py --tensorboard
$ tensorboard --logdir checkpoint/tensorboard
```
![110810440-78e25d80-828e-11eb-9689-523c4d12b772](https://user-images.githubusercontent.com/35613645/110831295-bfda4e00-82a2-11eb-9c04-b77b7a30fc4a.png)



##### Testing on WFLW test dataset
```
$ python3 test.py
```


##### Training 
Train on augumented WFLW dataset
```
$ python3 train.py
```



#### Folder structure    
    ├── model						# model's implementation
    ├── data						# data folder contains WFLW dataset & generated dataset
    	├── WFLW/					# extract WFLW images & annotations inside that folder
        	├── WFLW_annotations/
            ├── WFLW_images/
        ├── train					# generated train dataset
        ├── test					# generated test dataset


#### Refrences
MobileNet
- https://medium.com/analytics-vidhya/image-classification-with-mobilenet-cc6fbb2cd470
- https://medium.com/datadriveninvestor/review-on-mobile-net-v2-ec5cb7946784
- https://towardsdatascience.com/mobilenetv2-inverted-residuals-and-linear-bottlenecks-8a4362f4ffd5

3D-2D correspondences rotation:
- https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga549c2075fac14829ff4a58bc931c033d
- https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
- https://medium.com/analytics-vidhya/real-time-head-pose-estimation-with-opencv-and-dlib-e8dc10d62078

Other PFLD Implementations
- https://github.com/polarisZhao/PFLD-pytorch
- https://github.com/guoqiangqi/PFLD

Survey
- https://www.readcube.com/articles/10.1186%2Fs13640-018-0324-4

weak prespective projection
- https://www.cse.unr.edu/~bebis/CS791E/Notes/PerspectiveProjection.pdf
- https://en.wikipedia.org/wiki/3D_projection


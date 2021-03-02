# Practical Facial Landmarks Detector 

My unofficial implementation of [PFLD paper](https://arxiv.org/pdf/1902.10859.pdf) "Practical Facial Landmarks Detector" using Pytorch for a real time landmarks detection and head pose estimation.<br/>

![pfld](https://user-images.githubusercontent.com/35613645/109653302-89e1ef00-7b69-11eb-8dd7-e8810deebe44.png)


##### How to Install
```
 $ pip3 install -r requirements.txt
```
##### install opencv & dnn from source (optional)
Both opencv dnn & haar cascade are used for face detection, if you want to use haar cascade you can skip this part.
```
# Install minimal prerequisites (Ubuntu 18.04 as reference)
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
##### Tensorboard 
Take a wide look on dataset examples using tensorboard
```
$ python3 visualization.py --tensorboard
$ tensorboard --logdir checkpoint/tensorboard
```

##### Testing on WFLW test dataset
```
$ python3 test.py
```


##### Training 
Train on augumented WFLW dataset
```
$ python3 train.py
```

##### Testing on WFLW test dataset
```
$ python3 test.py
```


#### Refrences
Other PFLD Implementations
- https://github.com/polarisZhao/PFLD-pytorch
- https://github.com/guoqiangqi/PFLD

weak prespective projection
- https://www.cse.unr.edu/~bebis/CS791E/Notes/PerspectiveProjection.pdf
- https://en.wikipedia.org/wiki/3D_projection

3D-2D correspondences rotation:
- https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga549c2075fac14829ff4a58bc931c033d
- https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
- https://medium.com/analytics-vidhya/real-time-head-pose-estimation-with-opencv-and-dlib-e8dc10d62078

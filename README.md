# BU_EC601_MiniProject2

This the mini project 2 for BU course EC601  
I built a simple machine learning algorithm with TensorFlow to distinguish images of 2 projects: guitars and cellos  

The code was written in Python 3. Before running, please make sure that you have installed the following libraries: numpy, cv2, urllib, tensorflow

prepare_images.py is for downloading images from imagenet. I downloaded roughly 1000 pictures of guitars and cellos repectively. Some images downloaded showed no content. The function "find_uglies" is for deleting those images. After running the function "store_raw_images", please comment it. Make a directory called uglies and copy one image with no content inside. Then uncomment and run "find_uglies"

I manually chose the last 100 guitar and cello pictures as testing samples and the rest as training samples. Please create two directories called "train" adn "test", and two more directories called "guitars" and "cellos" in each of them. Save the pictures in corresponding directories and finally make a directory called "resized" in "train" and "test"  

Then please run train_test_tf.py to start the training and testing. It will call image_process.py, which transfer all the images into numpy arrays for training.

The testing result I got was around 75%.

Here are some references which I used:
https://www.wolfib.com/Image-Recognition-Intro-Part-1/
https://github.com/xuetsing/cats-vs-dogs
https://pythonprogramming.net/haar-cascade-object-detection-python-opencv-tutorial/

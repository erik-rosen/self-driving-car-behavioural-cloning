# **Behavioral Cloning** 

This project is part of Udacity's self driving car nanodegree. In this project I succesfully create and train a CNN that is able drive around a simulated track:

![alt text][image2]

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/angles.png "Labeled data"
[image2]: ./images/autonomous_driving.gif "Autonomous Driving"
[image3]: ./images/loss_plot.png "Loss Plot"
[image4]: ./images/architecture.png "Network Architecture"
[image5]: ./images/correction_factor.png "Computing correction factor"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to preprocess the labeled data and to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* run1.mp4 a full run in autonomous mode around the track using the trained model
* writeup.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model employs the same model as described in this [paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).

The input data is normalized in the model using a Keras lambda layer (code line 87) - this ensures that the input pixel values for the image are approximately zero mean and of unit variance. After that, we crop the input image to remove sections of the camera that capture scenery which may confuse the model and is not helful for navigation (the hood of the car, trees, mountains, and other terrain in the distance etc).

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting: Code line 32 in modely.py splits off the 4/5ths of labeled data into a training and a 1/5th to a validation set. Given that we have no hyperparameters that we are tuning, I chose not to use a test set. Plotting the MSE on the validation and the training set against the training epochs, we can see that the model actually performs better than the validation set than on the training set - indicating that there is no overfitting happening:

![alt text][image3]

On the basis of that, I chose not to introduce any dropout layers or regularization in the model as they simply were not necessary.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 101).

#### 4. Appropriate training data

Looking at the training data provided, we can see that the steering angle is mostly close to 0: 

![alt text][image1]

With training data where the steering angle is mostly 0, there is a risk that the model will not learn how to actually steer: a model which will always predict 0 (steer straight) would perform pretty well on this dataset. We will need to generate data that ensures that the model actually learns to recover. A model that knows how to drive in a straight line, but does not know how to recover if it gets off track is fairly useless. 

Note that there is no skewness in the distribution (the third central moment is close to 0), so generating "mirrored" training data is not necessary. To read more about how I went about genrating training data, see "Creation of the Training Set & Training Process" below.

### Model Architecture and Training Strategy

#### 1. Creation of the Training Set & Training Process

To make sure we had training data where the steering angle is less biased around 0, I chose to utilize the camera images from the left and right images in addition to the central camera (the car is capturing camera images from 3 forward facing cameras, one offset about 1 meter to the left from the center axis, one in the center and one offset 1 meter to the right). 

To make use of these images for training, we treat the left and right camera images as though they were captured from the center camera and apply a correction factor to the measured steering angle. The correction angle is computed by assuming that if the model sees an image that is 1 meter off the center lane axis, the model should aim to apply a steering output that gets back to the center after travelling 20 meters along the track.

![alt text][image5] 

This correction factor is computed in lines 35-40 in model.py.

For every sample used for training, we draw from the left, center or right camera with equal probability and apply the correction factor associated with the offset of the camera to the measurement. See lines 53-73 in model.py.

#### 2. Solution Design Approach

The overall strategy for deriving a model architecture was to base it off a model with known good performance for this task. 

My first step was to use a convolution neural network architecture similar to this [paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). 

Below is a figure displaying the architecture:

![alt text][image4]

The model is defined in lines 85-99 in model.py. The input data is normalized in the model using a Keras lambda layer (code line 87) - this ensures that the input pixel values for the image are approximately zero mean and of approximately unit variance. After that, we crop the input image to remove sections of the camera that capture scenery which may confuse the model and is not helful for navigation (the hood of the car, trees, mountains, and other terrain in the distance etc). 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. Code line 32 in model.py splits off the 4/5ths of labeled data into a training and a 1/5th to a validation set. Given that we have no hyperparameters that we are tuning, I chose not to use a test set. 

I initially chose to use batch gradient descent with a batch size of 32. The batches are loaded using a generator defined in lines 34-77 in model.py. The generator shuffles the dataset each epoch - see line 45. 

I trained the model using a mean squared error loss function and the adam optimizer (see lines 101-102 in model.py). Plotting the MSE on the validation and the training set against the training epochs (lines 113-121 in model.py), we can see that the model actually performs better on the validation set than on the training set - indicating that there is no overfitting happening. We also note that the loss of the validation set does not seem to improve much more past the second epoch, so we could have terminated training there.

![alt text][image3]

The final step was to run the simulator to see how well the car was driving around track one. The car successfully drove one full lap without leaving the road:

[image2]: ./images/autonomous_driving.gif "Autonomous Driving"

Full video of the lap can be seen [here](run1.mp4)
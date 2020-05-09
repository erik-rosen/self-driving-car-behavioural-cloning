# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

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

My model employs the same model as described in this [paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). Below is a figure displaying the architecture:

![alt text][image4]

The input data is normalized in the model using a Keras lambda layer (code line 87) - this ensures that the input pixel values for the image are approximately zero mean and of unit variance. After that, we crop the input image to remove sections of the camera that capture scenery which may confuse the model and is not helful for navigation (trees, mountains, and other terrain in the distance etc).

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting: Code line 32 in modely.py splits off the 4/5ths of labeled data into a training and a 1/5th to a validation set. Given that we have no hyperparameters that we are tuning, I chose not to use a test set. Plotting the MSE on the validation and the training set against the training epochs, we can see that the model actually performs better than the validation set than on the training set - indicating that there is no overfitting happening:

![alt text][image3]

On the basis of that, I chose not to introduce any dropout layers or regularization in the model as they simply were not necessary.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 101).

#### 4. Appropriate training data

Looking at the training data provided, we can see that the steering angle is mostly close to 0: 

![alt text][image1]

We will need to generate data that teaches the car to actually recover if it needs to. A model that knows how to drive in a straight line, but does not know how to recover if it gets off track is fairly useless. To read more about how I went about genrating training data, go to "Creation of the Training Set & Training Process" below.

### Model Architecture and Training Strategy

#### 1. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

#### 2. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 3. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.

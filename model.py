import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from math import ceil, atan, degrees
from scipy import ndimage
import numpy as np
import sklearn as sk
import random
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D

samples = []
angles = []
with open('../data/driving_log.csv') as csv_file:
    reader = csv.reader(csv_file)
    for line in reader:
        if line[0].endswith('.jpg') and line[1].endswith('.jpg') and line[2].endswith('.jpg'):
            samples.append(line)
            angles.append(float(line[3]))

# Plot steering angles of labeled dataset
num_bins = 100
n, bins, patches = plt.hist(angles, num_bins, facecolor='blue', alpha=0.5)
plt.title('Histogram of steering angle of training images')
plt.ylabel('Number of occurences')
plt.xlabel('Angle')
plt.savefig('angles.png')

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    # Below computes the correction steering angles for the left and right camera images
    correction_distance = 20 # how many meters the car needs to travel with the corrected steering to get to the desired destination
    camera_offset = 1 # in meters
    max_angle = 45 # in degrees - when the output measurement reads 1 or -1, the wheel angle is assumed to be 45 degrees from the center axis
    correction_angle = degrees(atan(camera_offset/correction_distance))
    correction_factor = correction_angle/max_angle

    num_samples = len(samples)
    while True:
        
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            measurements = []
            for batch_sample in batch_samples:
                # Below pairs up camera image paths and corrected measurements
                camera_options = [
                    {"camera":'left', "measurement": float(batch_sample[3]) + correction_factor, "source_path": batch_sample[1]}, 
                    {"camera":'center', "measurement": float(batch_sample[3]), "source_path": batch_sample[0]}, 
                    {"camera":'right', "measurement": float(batch_sample[3]) - correction_factor, "source_path": batch_sample[2]} 
                ]
                # Draw left, center or right image with associated corrected measurements
                camera = \
                    camera_options[\
                        random.randrange(len(camera_options))\
                    ]
                
                # Correct the path for the local environment
                source_path = camera["source_path"]
                filename = source_path.split('/')[-1]
                current_path = '../data/IMG/' + filename
                image = ndimage.imread(current_path)
                
                #Ensure that the image file has the right shape
                if image.shape == (160,320,3):
                    images.append(image)
                    measurements.append(camera["measurement"])
            
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sk.utils.shuffle(X_train, y_train)

batch_size=32

train_generator = generator(train_samples, batch_size)
validation_generator = generator(validation_samples, batch_size)

# Setting up network as described here: http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
model = Sequential()
# Normalizing input (appriximately zero mean, unit variance)
model.add(Lambda(lambda x: x/255 - 0.5, input_shape=(160,320,3)))
# Crop top and bottom of camera image which can distract the model
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, \
            steps_per_epoch=ceil(len(train_samples)/batch_size), \
            validation_data=validation_generator, \
            validation_steps=ceil(len(validation_samples)/batch_size), \
            epochs=5, verbose=1)

model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
#plt.show()
plt.savefig('loss_plot.png')
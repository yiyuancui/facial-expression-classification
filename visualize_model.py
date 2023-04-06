import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.svm import SVC
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers import UpSampling2D
from keras.optimizers import Adam
from keras.layers import BatchNormalization
from keras import regularizers
from keras.utils.vis_utils import plot_model

X_train = np.load('X_train_total.npy', allow_pickle=True)


model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:]))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2), name='pool1'))
model.add(Dropout(0.30))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.20))

model.add(Conv2D(96, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.30))

model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(32, activation='relu', name='dense1'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Dense(8, activation='softmax', name='dense2'))

# Plot the model architecture
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
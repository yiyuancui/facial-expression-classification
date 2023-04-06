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
from tensorflow.keras import backend as K
def custom_precision(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def custom_recall(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall
def custom_f1_score(y_true, y_pred):
    precision = custom_precision(y_true, y_pred)
    recall = custom_recall(y_true, y_pred)
    f1_score = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_score


X_train = np.load('X_train_total.npy', allow_pickle=True)
X_val = np.load('X_val_total.npy', allow_pickle=True)
y_train = np.load('y_train_total.npy', allow_pickle=True)
y_val = np.load('y_val_total.npy', allow_pickle=True)

# Load the trained model from the saved file
model = load_model('complex_cnn_test_0.7.h5', custom_objects={
    'custom_precision': custom_precision,
    'custom_recall': custom_recall,
    'custom_f1_score': custom_f1_score
})

# Evaluate the model on the validation data
with tf.device('/GPU:0'):
    loss, acc, f1_score, precision, recall = model.evaluate(X_val, y_val,verbose=1)

print('Test loss:', loss)
print('Test accuracy:', acc)
print('Test F1 score:', f1_score)
print('Test precision:', precision)
print('Test recall:', recall)

print("Validation loss:", loss)
print("Validation accuracy:", acc)

plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)



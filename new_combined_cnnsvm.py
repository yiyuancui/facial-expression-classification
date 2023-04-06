import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.svm import SVC
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers import UpSampling2D
from keras.optimizers import Adam
from keras.layers import BatchNormalization
from keras import regularizers
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K


X_train = np.load('X_train_total.npy', allow_pickle=True)
X_val = np.load('X_val_total.npy', allow_pickle=True)
y_train = np.load('y_train_total.npy', allow_pickle=True)
y_val = np.load('y_val_total.npy', allow_pickle=True)
optimizer = Adam(learning_rate=0.001)

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



with tf.device('/GPU:0'):  # Use GPU for faster training
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:]))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), name='pool1'))
    model.add(Dropout(0.4))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Conv2D(96, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.30))

    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(32, activation='relu', name='dense1'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(8, activation='softmax', name='dense2'))
    model.summary()
    # Compile the CNN model with appropriate loss and optimizer
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy',custom_f1_score,
                       custom_precision,
                       custom_recall])

    history = model.fit(X_train, y_train, epochs=40, batch_size=32, validation_data=(X_val, y_val))
model.save('complex_cnn_test_0.7.h5')
with tf.device('/GPU:0'):
    loss, acc, f1_score, precision, recall = model.evaluate(X_val, y_val,verbose=1)

print('Test loss:', loss)
print('Test accuracy:', acc)
print('Test F1 score:', f1_score)
print('Test precision:', precision)
print('Test recall:', recall)

# Plot the training and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show()

# Plot the training and validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='lower right')
plt.show()


plt.plot(history.history['custom_f1_score'], label='Training F1 Score')
plt.plot(history.history['val_custom_f1_score'], label='Validation F1 Score')
plt.legend()
plt.show()

plt.plot(history.history['custom_precision'], label='Training Precision')
plt.plot(history.history['val_custom_precision'], label='Validation Precision')
plt.legend()
plt.show()

plt.plot(history.history['custom_recall'], label='Training Recall')
plt.plot(history.history['val_custom_recall'], label='Validation Recall')
plt.legend()
plt.show()
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
import cv2
import pandas as pd

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
# Load your trained CNN model
model = load_model('h5 files/complex_cnn_test_0.7.h5', custom_objects={
    'custom_precision': custom_precision,
    'custom_recall': custom_recall,
    'custom_f1_score': custom_f1_score
})

#anger:'anger/image0000061.jpg'

# Load and preprocess the input image
img_path = 'anger/image0000294.jpg'
image_data = []
img = cv2.imread(img_path)
if img is None:
    print(f"Image file not found or unable to read: {img_path}")
else:
    img = cv2.resize(img, (144, 144))
    img = img / 255.0
    image_data.append(img)

image_data = np.array(image_data)
x = image_data
x_tensor = tf.convert_to_tensor(x)
output = model(x_tensor)

with tf.GradientTape() as tape:
    tape.watch(x_tensor)
    output = model(x_tensor)
grads = tape.gradient(output, x_tensor)

# # Calculate the saliency map
# saliency_map = np.max(np.abs(grads), axis=-1)[0]
#
# epsilon = 1e-5
# saliency_map = (saliency_map - np.min(saliency_map)) / (np.max(saliency_map) - np.min(saliency_map) + epsilon)
# Calculate the saliency map
saliency_map = np.max(np.abs(grads), axis=-1)[0]

# Use percentile-based normalization
lower_percentile = 1
upper_percentile = 99
lower = np.percentile(saliency_map, lower_percentile)
upper = np.percentile(saliency_map, upper_percentile)

e = 1e-8
saliency_map = np.clip(saliency_map, lower, upper)
saliency_map = (saliency_map - lower) / (upper - lower+e)


saliency_map_blurred = cv2.GaussianBlur(saliency_map, (5, 5), 0)


saliency_map_stretched = np.zeros_like(saliency_map_blurred)
saliency_map_stretched = cv2.normalize(saliency_map_blurred, saliency_map_stretched, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

predicted_class_index = np.argmax(output)


class_names = ['surprise', 'anger', 'fear', 'disgust', 'sad', 'neutral', 'contempt', 'happy']
predicted_class_name = class_names[predicted_class_index]

print(f"Predicted class index: {predicted_class_index}")
print(f"Predicted class name: {predicted_class_name}")



img2 = cv2.imread(img_path)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.imshow(img2)
ax1.set_title('Original Image')
ax2.imshow(saliency_map, cmap='hot')
ax2.set_title('Saliency Map')
ax3.imshow(saliency_map_stretched, cmap='hot')
ax3.set_title('Stretched Saliency Map')
plt.show()

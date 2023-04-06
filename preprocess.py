import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np

# Load the CSV file into a Pandas DataFrame
df = pd.read_csv('labels.csv')
df['label'] = df['label'].replace(['surprise'], 0)
df['label'] = df['label'].replace(['anger'], 1)
df['label'] = df['label'].replace(['fear'], 2)
df['label'] = df['label'].replace(['disgust'], 3)
df['label'] = df['label'].replace(['sad'], 4)
df['label'] = df['label'].replace(['neutral'], 5)
df['label'] = df['label'].replace(['contempt'], 6)
df['label'] = df['label'].replace(['happy'], 7)

# Preprocess the image data and create a list of image arrays
image_data = []
for filepath in df['pth']:
    img = cv2.imread(filepath)
    # Resize and preprocess the image as needed
    img = cv2.resize(img, (244, 244))
    img = img / 255.0 # normalize the pixel values to be between 0 and 1
    image_data.append(img)

# Merge the labels with the image data and perform one-hot encoding on the labels
image_data = np.array(image_data)
labels = df['label']
one_hot_labels = to_categorical(labels)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(image_data, one_hot_labels, test_size=0.2, random_state=42)

np.save('X_train_total_244.npy', X_train)
np.save('X_val_total_244.npy', X_val)
np.save('y_train_total_244.npy', y_train)
np.save('y_val_total_244.npy', y_val)
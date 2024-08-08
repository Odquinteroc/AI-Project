import random
import pandas as pd
import numpy as np
import os, cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils import shuffle
from google.colab import drive
import zipfile
import os
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.callbacks import TensorBoard
from google.colab import files
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix

# Define the list of fruits
fruits = ["Apple", "Banana", "Bellpepper", "Carrot", "Cucumber", "Grape", "Guava", "Jujube", "Mango", "Orange", "Pomegranate", "Potato", "Strawberry", "Tomato"]

# Paths for random images and output directory
random_images = f"/content/drive/MyDrive/IA_proyect/Random/data"
output_path = f"/content/drive/MyDrive/IA_proyect/output/augmentation/fruit_or_not/"
os.makedirs(output_path, exist_ok=True)

# Define function to calculate TP, FP, FN, TN
def calculate_metrics(y_true, y_pred):
    # Get the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Extract TP, FP, FN, TN
    TP = cm[1, 1]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TN = cm[0, 0]

    return TP, FP, FN, TN

# Parameters for loading data
image_size = 128
batch_size_val = 60
epochs_val = 3
X = []
y = []

# Initialize labels: 1 if fruit, 0 if not fruit
label = 1

# Iterate over each fruit
for fruit in fruits:
    # Define input and output paths
    input_path = f"/content/drive/MyDrive/IA_proyect/{fruit}/ex_images"

    # List of categories
    categories = os.listdir(input_path)

    for category in categories:
        print('Loading --> ', category, '--> Label', str(label))

        # Path to the category folder
        category_path = os.path.join(input_path, category)

        # List all files in the specified directory
        files_images = os.listdir(category_path)

        # Randomly select 165 files from the list
        files_images = random.sample(files_images, 165)
        print(len(files_images))
        for img_name in files_images:
            img_path = os.path.join(category_path, img_name)

            # Load the image using OpenCV
            image = cv2.imread(img_path)

            # Check if the image loads correctly
            if image is None:
                print(f"Error loading image: {img_name}")
                continue

            # Resize the image
            image = cv2.resize(image, (image_size, image_size))

            # Add the image and label to the lists
            X.append(image)
            y.append(label)

# Initialize labels: 1 if fruit, 0 if not fruit
label = 0

# List of random images
files_images = os.listdir(random_images)

for img_name in files_images:
    img_path = os.path.join(random_images, img_name)

    # Load the image using OpenCV
    image = cv2.imread(img_path)

    # Check if the image loads correctly
    if image is None:
        print(f"Error loading image: {img_name}")
        continue

    # Resize the image
    image = cv2.resize(image, (image_size, image_size))

    # Add the image and label to the lists
    X.append(image)
    y.append(label)

# Normalize the data to a 0 to 1 scale
X = np.array(X).astype(float)/255
y = np.array(y)

print("The shape of X is: ", X.shape)
print("The shape of y is: ", y.shape)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=15,
    horizontal_flip=True,
    vertical_flip=True
)

datagen.fit(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

data_gen_training = datagen.flow(X_train, y_train, batch_size=100)

modelCNN_color = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(image_size, image_size, 3)),  # Explicitly define input shape
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(250, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

modelCNN_color.compile(optimizer='adam',
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])

print(f"Training X set size: {len(X_train)}")
print(f"Test X set size: {len(X_test)}")
print(f"Training y set size: {len(y_train)}")
print(f"Test y set size: {len(y_test)}")

TensorBoardCNN_color = TensorBoard(log_dir='logs/cnn_color')

modelCNN_color.fit(data_gen_training, batch_size=500, epochs=50,
                   validation_data=(X_test, y_test),
                   steps_per_epoch=int(np.ceil(len(X_train)/float(32))),
                   validation_steps=int(np.ceil(len(X_test)/float(32))),
                   callbacks=[TensorBoardCNN_color])

# Plot training and validation loss
plt.figure(figsize=(10, 10))
plt.xlabel("# Epochs")
plt.ylabel("Loss")
plt.plot(modelCNN_color.history.history['loss'], label='Training Loss')
plt.plot(modelCNN_color.history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.savefig(f"{output_path}loss_plot.png")

# Save the model
modelCNN_color.save(f"{output_path}modelCNN_color.h5")

# Predictions on the test set
y_pred_probs = modelCNN_color.predict(X_test)
y_estimado = np.argmax(y_pred_probs, axis=1)

# Predictions on the training set
y_pred_probs = modelCNN_color.predict(X_train)
y_estimado_train = np.argmax(y_pred_probs, axis=1)

# Contingency table using the confusion matrix
contingency_table = confusion_matrix(y_test, y_estimado)
contingency_table
# Convert to DataFrame for better visualization

# Calculate metrics for the test set
TP_test, FP_test, FN_test, TN_test = calculate_metrics(y_test, y_estimado)

# Calculate metrics for the training set
TP_train, FP_train, FN_train, TN_train = calculate_metrics(y_train, y_estimado_train)

# Create a DataFrame to store the results
results_df = pd.DataFrame(
    [
        {"Fruta": "Is fruit or not", "Label_value 0": "not fruit", "Label_value 1": "fruit", "TP": TP_test, "FP": FP_test, "FN": FN_test, "TN": TN_test, "Dataset": "Test", "Label Positive": 1, "Label Negative": 0},
        {"Fruta": "Is fruit or not", "Label_value 0": "not fruit", "Label_value 1": "fruit", "TP": TP_train, "FP": FP_train, "FN": FN_train, "TN": TN_train, "Dataset": "Train", "Label Positive": 1, "Label Negative": 0}
    ]
)

# Define the file path where you want to save the CSV
csv_file_path = f"{output_path}metrics.csv"

# Save the DataFrame to a CSV file with a semicolon separator
results_df.to_csv(csv_file_path, sep=';', index=False, header=True)

# Define the paths
keras_model_path = f"{output_path}modelCNN_color.h5"  # Path to your Keras model
output_folder = f"{output_path}js/"  # Directory to store the converted model
os.makedirs(output_folder, exist_ok=True)

# Run the conversion
# !tensorflowjs_converter --input_format keras {keras_model_path} {output_folder}

print(f"Model converted and saved to {output_folder}")

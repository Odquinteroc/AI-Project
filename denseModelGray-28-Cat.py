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
import random

# Define the list of fruits
fruits = ["Apple","Banana","Bellpepper","Carrot","Cucumber","Grape","Guava","Jujube","Mango","Orange","Pomegranate","Potato","Strawberry","Tomato"]

# Define the output path where results will be stored
output_path = f"/content/drive/MyDrive/IA_proyect/output/dense_grayscale_without_aumentation/1_model_28_categories/"
os.makedirs(output_path, exist_ok=True)

def sample_images(files_images, num_samples=500):
    """
    Displays a fixed number of images from a list, using replacement if necessary.

    Parameters:
    - files_images: list of image files.
    - num_samples: number of required samples (default 500).

    Returns:
    - A list of selected images.
    """
    # If the number of files is greater than or equal to the required number of samples,
    # use sampling without replacement
    if len(files_images) >= num_samples:
        sampled_images = random.sample(files_images, num_samples)
    else:
        # If there are fewer files than the required number of samples,
        # use sampling with replacement
        sampled_images = random.choices(files_images, k=num_samples)

    return sampled_images

def calculate_metrics_all_categories(y_true, y_pred, num_classes=28):
    # Dictionary to store metrics for each category
    metrics = {category: {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0} for category in range(0, num_classes)}

    # Calculate the confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(0, num_classes))

    for category in range(0, num_classes):
        # TP: True Positives
        TP = cm[category-1, category-1]

        # FP: False Positives (sum of the current column, except TP)
        FP = cm[:, category-1].sum() - TP

        # FN: False Negatives (sum of the current row, except TP)
        FN = cm[category-1, :].sum() - TP

        # TN: True Negatives (total sum minus the sum of the current row and column, plus TP)
        TN = cm.sum() - (TP + FP + FN)

        # Store the metrics in the dictionary
        metrics[category]['TP'] = TP
        metrics[category]['TN'] = TN
        metrics[category]['FP'] = FP
        metrics[category]['FN'] = FN

    # Convert the dictionary to a DataFrame
    metrics = pd.DataFrame.from_dict(metrics, orient='index')

    # Rename the index to be a column with the name 'label'
    metrics = metrics.reset_index().rename(columns={'index': 'label'})

    return metrics

# Define a function to calculate TP, FP, FN, TN
def calculate_metrics(y_true, y_pred):
  # Get the confusion matrix
  cm = confusion_matrix(y_true, y_pred)

  # Extract TP, FP, FN, TN
  TP = cm[1, 1]
  FP = cm[0, 1]
  FN = cm[1, 0]
  TN = cm[0, 0]

  return TP, FP, FN, TN

# Parameters for data loading
image_size = 128
batch_size_val = 60
epochs_val = 3
X = []
y = []

# To store which label corresponds to each category
data = []
# Initialize labels: 1 if fruit, 0 if not fruit
label = 0

# Iterate over each fruit
for fruit in fruits:
  # Define input and output paths
  input_path = f"/content/drive/MyDrive/IA_proyect/{fruit}/ex_images"

  # List of categories
  categories = os.listdir(input_path)

  for category in categories:

    data.append((category, label))
    print('Loading --> ', category, '--> Label', str(label))

    # Path to the category folder
    category_path = os.path.join(input_path, category)

    # List all files in the specified directory
    files_images = os.listdir(category_path)

    # Randomly select 500 files from the list
    files_images = sample_images(files_images, num_samples=500)
    print(len(files_images))
    for img_name in files_images:
      img_path = os.path.join(category_path, img_name)

      # Load the image using OpenCV
      image = cv2.imread(img_path)

      # Check if the image is loaded correctly
      if image is None:
        print(f"Error loading image: {img_name}")
        continue

      # Resize the image
      image = cv2.resize(image, (image_size, image_size))

      # Convert the image to grayscale
      image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      
      # Change the shape to 128, 128, 1
      image = image.reshape(image_size, image_size, 1)

      # Add the image and the label to the lists
      X.append(image)
      y.append(label)
    label = label + 1

# Create a DataFrame with the columns 'category' and 'label'
categoria_label = pd.DataFrame(data, columns=['categoria', 'label'])

# Define the file path where you want to save the CSV
csv_file_path1 = f"{output_path}categoria_label.csv"

# Save the DataFrame to a CSV file with a semicolon separator
categoria_label.to_csv(csv_file_path1, sep=';', index=False, header=True)

# Normalization to scale 0 to 1
X = np.array(X).astype(float)/255
y = np.array(y)

print("The shape of X is: ", X.shape)
print("The shape of y is: ", y.shape)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

# Define the model architecture
model_dense = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(image_size,image_size,1)), # Flatten the matrix to 1D with 10000 neurons where each pixel will be received
    tf.keras.layers.Dense(150, activation='relu'),# Hidden dense layer with 150 neurons
    tf.keras.layers.Dense(150, activation='relu'), # Hidden dense layer with 150 neurons
    tf.keras.layers.Dense(28, activation='softmax'), # Output layer with 28 neurons using softmax activation (sum of outputs = 1)
])

# Compile the model
model_dense.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

print(f"Training X set size: {len(X_train)}")
print(f"Test X set size: {len(X_test)}")
print(f"Training y set size: {len(y_train)}")
print(f"Test y set size: {len(y_test)}")

# Define a TensorBoard callback
TensorBoardCNN_color = TensorBoard(log_dir='logs/cnn_dense')

# Train the model
model_dense.fit(X_train, y_train, batch_size= 500, epochs=50,
            validation_data = (X_test, y_test),
            steps_per_epoch =int(np.ceil(len(X_train)/float(32))),
            validation_steps =int(np.ceil(len(X_test)/float(32))),
            callbacks=[TensorBoardCNN_color])

# Plot the training and validation loss
plt.figure(figsize=(10, 10))
plt.xlabel("# Epochs")
plt.ylabel("Loss")
plt.plot(model_dense.history.history['loss'], label='Training Loss')
plt.plot(model_dense.history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.savefig(f"{output_path}loss_plot.png")

# Save the model
model_dense.save(f"{output_path}modelCNN_color.h5")

# Predictions on the test set
y_pred_probs = model_dense.predict(X_test)
y_estimado = np.argmax(y_pred_probs, axis=1)

# Predictions on the training set
y_pred_probs = model_dense.predict(X_train)
y_estimado_train = np.argmax(y_pred_probs, axis=1)

# Calculate metrics for the test set
metrics_test = calculate_metrics_all_categories(y_test, y_estimado)

# Calculate metrics for the training set
metrics_train = calculate_metrics_all_categories(y_train, y_estimado_train)

# Define the file path where you want to save the CSV
csv_file_path = f"{output_path}metrics_test.csv"

# Save the DataFrame to a CSV file with a semicolon separator
metrics_test.to_csv(csv_file_path, sep=';', index=False, header=True)

# Define the file path where you want to save the CSV
csv_file_path = f"{output_path}metrics_train.csv"

# Save the DataFrame to a CSV file with a semicolon separator
metrics_train.to_csv(csv_file_path, sep=';', index=False, header=True)

# Define the paths
keras_model_path = f"{output_path}modelCNN_color.h5"  # Path to your Keras model
output_folder = f"{output_path}js/"   # Directory to store the converted model
os.makedirs(output_folder, exist_ok=True)

# Run the conversion to TensorFlow.js format
# !tensorflowjs_converter --input_format keras {keras_model_path} {output_folder}

print(f"Model converted and saved to {output_folder}")

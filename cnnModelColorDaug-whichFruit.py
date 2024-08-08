import os
import random
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import TensorBoard

# Configuration
FRUITS = ["Banana", "Carrot", "Grape", "Guava", "Jujube", "Orange", "Pomegranate", "Strawberry", "Tomato"]
OUTPUT_PATH = "/content/drive/MyDrive/IA_proyect/output/color_augmentation/2_model_14_categories/"
IMAGE_SIZE = 128
BATCH_SIZE = 500
EPOCHS = 100
NUM_CLASSES = 9

os.makedirs(OUTPUT_PATH, exist_ok=True)

def sample_images(files_images, num_samples=500):
    """Samples a fixed number of images from a list, using replacement if necessary."""
    return random.choices(files_images, k=num_samples)

def calculate_metrics_all_categories(y_true, y_pred, num_classes=NUM_CLASSES):
    """Calculates metrics for all categories."""
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    metrics = {category: {} for category in range(num_classes)}

    for category in range(num_classes):
        TP = cm[category, category]
        FP = cm[:, category].sum() - TP
        FN = cm[category, :].sum() - TP
        TN = cm.sum() - (TP + FP + FN)

        metrics[category] = {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN}

    return pd.DataFrame.from_dict(metrics, orient='index').reset_index().rename(columns={'index': 'label'})

def load_and_preprocess_data():
    """Loads and preprocesses image data."""
    X, y = [], []
    data = []

    for label, fruit in enumerate(FRUITS):
        input_path = f"/content/drive/MyDrive/IA_proyect/{fruit}/ex_images"
        categories = os.listdir(input_path)

        for category in categories:
            data.append((category, label))
            print(f'Loading --> {category} --> Label {label}')

            category_path = os.path.join(input_path, category)
            files_images = sample_images(os.listdir(category_path))

            for img_name in files_images:
                img_path = os.path.join(category_path, img_name)
                image = cv2.imread(img_path)

                if image is None:
                    print(f"Error loading image: {img_name}")
                    continue

                image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
                X.append(image)
                y.append(label)

    category_label = pd.DataFrame(data, columns=['category', 'label'])
    X = np.array(X).astype(float) / 255
    y = np.array(y)

    return X, y, category_label

def create_model():
    """Creates and compiles the CNN model."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(250, activation='relu'),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Load and preprocess data
X, y, category_label = load_and_preprocess_data()
print(f"Shape of X: {X.shape}, Shape of y: {y.shape}")

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=30, width_shift_range=0.2, height_shift_range=0.2,
    shear_range=15, horizontal_flip=True, vertical_flip=True
)
datagen.fit(X_train)

# Create and train model
model = create_model()
tensorboard_callback = TensorBoard(log_dir='logs/cnn_color')

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    epochs=EPOCHS,
    validation_data=(X_test, y_test),
    steps_per_epoch=int(np.ceil(len(X_train) / float(BATCH_SIZE))),
    validation_steps=int(np.ceil(len(X_test) / float(BATCH_SIZE))),
    callbacks=[tensorboard_callback]
)

# Save the model and loss plot
model.save(f"{OUTPUT_PATH}modelCNN_color.h5")

plt.figure(figsize=(10, 10))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig(f"{OUTPUT_PATH}loss_plot.png")

# Calculate and save metrics
y_pred_test = np.argmax(model.predict(X_test), axis=1)
y_pred_train = np.argmax(model.predict(X_train), axis=1)

metrics_test = calculate_metrics_all_categories(y_test, y_pred_test)
metrics_train = calculate_metrics_all_categories(y_train, y_pred_train)

metrics_test.to_csv(f"{OUTPUT_PATH}metrics_test.csv", sep=';', index=False)
metrics_train.to_csv(f"{OUTPUT_PATH}metrics_train.csv", sep=';', index=False)

# Convert the model to TensorFlow.js format
keras_model_path = f"{OUTPUT_PATH}modelCNN_color.h5"
output_folder = f"{OUTPUT_PATH}js/"
os.makedirs(output_folder, exist_ok=True)

os.system(f"tensorflowjs_converter --input_format keras {keras_model_path} {output_folder}")
print(f"Model converted and saved to {output_folder}")

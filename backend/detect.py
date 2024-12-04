import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.model_selection import train_test_split
import os
import cv2
import numpy as np

# Load data function
def load_data(data_dir, image_size=128):
    images, labels = [], []
    for filename in os.listdir(data_dir):
        if filename.endswith(".png"):
            filepath = os.path.join(data_dir, filename)
            label = filename.split("_")[0]
            img = cv2.imread(filepath)
            img = cv2.resize(img, (image_size, image_size))
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

# Label mapping
label_mapping = {"circle": 0, "square": 1, "triangle": 2}

# Load and preprocess data
images, labels = load_data("shapes_dataset")
labels = np.array([label_mapping[label] for label in labels])
images = images / 255.0  # Normalize pixel values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# One-hot encode labels
y_train = tf.keras.utils.to_categorical(y_train, num_classes=3)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=3)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=360,
    width_shift_range=0.5,
    height_shift_range=0.5,
    shear_range=0.5,
    zoom_range=0.5,
    brightness_range=[0.5, 1.5],
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)
datagen.fit(X_train)

# Load pretrained MobileNetV2
base_model = MobileNetV2(weights="/Users/dallas/models/shape_detection_model.h5", include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False  # Freeze the base model layers

# Define the model
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(3, activation="softmax")  # Output layer for 3 classes
])

# Compile the model with a smaller learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])

# Learning rate scheduler
def scheduler(epoch, lr):
    if epoch > 5:
        return lr * 0.1
    return lr

lr_scheduler = LearningRateScheduler(scheduler)

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)

# Train the model
model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=50,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, lr_scheduler]
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.2f}")

model.save("shape_detection_model.h5")
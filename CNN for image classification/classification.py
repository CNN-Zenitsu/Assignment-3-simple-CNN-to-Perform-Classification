# Simple binary classification CNN for waste detection
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Path to training images
train_dir = "dataset/Thumbnail images/Real images/"

# Data preprocessing with normalization and validation split
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Training data generator
train_gen = datagen.flow_from_directory(
    train_dir,
    target_size=(128,128),  # Resize images to 128x128
    batch_size=32,
    class_mode="binary",  # Binary classification (waste/no-waste)
    subset="training"
)

# Validation data generator
val_gen = datagen.flow_from_directory(
    train_dir,
    target_size=(128,128),
    batch_size=32,
    class_mode="binary",
    subset="validation"
)

# Simple CNN architecture for binary classification
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    tf.keras.layers.MaxPooling2D(),  # Reduce spatial dimensions
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),  # Convert to 1D
    tf.keras.layers.Dense(64, activation='relu'),  # Hidden layer
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary output
])

# Compile and train model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(train_gen, validation_data=val_gen, epochs=30)


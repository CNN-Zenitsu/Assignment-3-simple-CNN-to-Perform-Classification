# Split dataset into train/validation/test sets with normalization
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
import numpy as np
from tensorflow.keras.optimizers import Adam

# Dataset path
data_dir = "/content/RealWaste"

# Image and training parameters
img_height, img_width = 128, 128
batch_size = 32
seed = 123  # For reproducible results

# Step 1: Initial split - 70% train, 30% temp (for val+test)
train_ds_raw = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.3,   # 30% goes to temp
    subset="training",
    seed=seed,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

temp_ds_raw = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.3,
    subset="validation",    # the 30% "temp" set
    seed=seed,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Extract class names and count
class_names = train_ds_raw.class_names
num_classes = len(class_names)
print("Classes:", class_names)

# Step 2: Split temp set into validation (15%) and test (15%)
val_size = 0.5  # Half of temp_ds (0.15 / 0.30 = 0.5)
val_ds_raw = temp_ds_raw.take(int(len(temp_ds_raw) * val_size))
test_ds_raw = temp_ds_raw.skip(int(len(temp_ds_raw) * val_size))

# Step 3: Normalize pixel values from [0,255] to [0,1]
normalization_layer = tf.keras.layers.Rescaling(1./255)

train_ds = train_ds_raw.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds_raw.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds_raw.map(lambda x, y: (normalization_layer(x), y))

# Optimize dataset performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)  # Cache, shuffle, prefetch
val_ds = val_ds.cache().prefetch(AUTOTUNE)
test_ds = test_ds.cache().prefetch(AUTOTUNE)

print("\n✅ Dataset Splits:")
print(f"Training batches: {len(train_ds)}")
print(f"Validation batches: {len(val_ds)}")
print(f"Testing batches: {len(test_ds)}")



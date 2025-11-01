# CNN model architecture for waste classification
model = models.Sequential([
    # First Conv Block - Basic feature detection
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.BatchNormalization(),  # Stabilizes training
    layers.Conv2D(32, (3, 3), activation='relu'),  # Extract more features
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),  # Downsample by 2x
    layers.Dropout(0.25),  # Prevent overfitting

    # Second Conv Block - More complex patterns
    layers.Conv2D(64, (3, 3), activation='relu'),  # Double filters
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    # Third Conv Block - High-level features
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    # Classification head
    layers.Flatten(),  # Convert 2D to 1D
    layers.Dense(256, activation='relu'),  # Feature combination
    layers.BatchNormalization(),
    layers.Dropout(0.5),  # Strong regularization
    layers.Dense(num_classes, activation='softmax')  # Output probabilities
])
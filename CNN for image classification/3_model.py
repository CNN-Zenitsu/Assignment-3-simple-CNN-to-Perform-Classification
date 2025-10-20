model = models.Sequential([
    # First Conv Block
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.BatchNormalization(),  # Stabilizes training and improves convergence
    layers.Conv2D(32, (3, 3), activation='relu'),  # Additional Conv layer for better feature extraction
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),  # Reduces spatial dimensions
    layers.Dropout(0.25),  # Prevents overfitting

    # Second Conv Block
    layers.Conv2D(64, (3, 3), activation='relu'),  # Increased filters
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu'),  # Additional Conv layer
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    # Third Conv Block (optional for deeper feature extraction)
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    # Dense Layers
    layers.Flatten(),
    layers.Dense(256, activation='relu'),  # Increased units
    layers.BatchNormalization(),
    layers.Dropout(0.5),  # Higher dropout to prevent overfitting
    layers.Dense(num_classes, activation='softmax')  # Output layer
])
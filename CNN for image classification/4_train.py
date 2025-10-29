# Training configuration and model compilation
optimizer = Adam(learning_rate=0.001)  # Adam optimizer with standard learning rate

# Compile model with appropriate loss function for multi-class classification
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',  # For integer labels
              metrics=['accuracy'])

model.summary()

# Train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20  # Number of training iterations
)

# Evaluate model performance
test_loss, test_acc = model.evaluate(test_ds)
print(f"\n✅ Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Visualize training progress - Accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()

# Visualize training progress - Loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# Final evaluation on all sets
train_loss, train_acc = model.evaluate(train_ds)
print(f"Training Accuracy: {train_acc:.4f}")

test_loss, test_acc = model.evaluate(test_ds)
print(f"Test Accuracy: {test_acc:.4f}")

# Generate predictions for detailed analysis
y_true = np.concatenate([y for x, y in test_ds], axis=0)  # True labels
y_pred_probs = model.predict(test_ds)  # Prediction probabilities
y_pred = np.argmax(y_pred_probs, axis=1)  # Predicted classes

# Confusion matrix visualization
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues', xticks_rotation='vertical')
plt.title("Confusion Matrix")
plt.show()

# Detailed classification metrics
from sklearn.metrics import classification_report

report = classification_report(y_true, y_pred, digits=4)
print(report)

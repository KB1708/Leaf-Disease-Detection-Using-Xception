import tensorflow as tf
import os
import json
import matplotlib.pyplot as plt
import argparse

# --- 1. Argument Parsing for Local Execution ---
# This allows us to run the script from the command line and easily change settings.
parser = argparse.ArgumentParser(description='Train a Leaf Disease Detection Model.')
parser.add_argument('--data_dir', type=str, required=True, help='Path to the combined dataset directory (PlantVillage-Combined).')
parser.add_argument('--model_output_dir', type=str, default='../../models/', help='Directory to save the trained model and class names.')
parser.add_argument('--img_size', type=int, default=224, help='Image size (height and width).')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training. Default is 16 to be safe on memory.')
parser.add_argument('--epochs', type=int, default=25, help='Number of epochs to train for.')
args = parser.parse_args()


# --- 2. Modified Data Loading & Performance ---
# This section has been changed to correctly create a validation split.
print("🌿 Loading and preparing datasets...")

# Create the training dataset from 80% of the images in the 'train' directory.
train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(args.data_dir, 'train'),
    validation_split=0.2, # NEW: Specify the split size
    subset="training",      # NEW: Specify this is the training set
    seed=123,               # NEW: Add a seed for reproducibility
    image_size=(args.img_size, args.img_size),
    batch_size=args.batch_size
)

# Create the validation dataset from the remaining 20% of the images.
val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(args.data_dir, 'train'), # We point to the same 'train' directory
    validation_split=0.2, # NEW: Same split size
    subset="validation",    # NEW: Specify this is the validation set
    seed=123,               # NEW: Same seed to ensure no overlap
    image_size=(args.img_size, args.img_size),
    batch_size=args.batch_size
)


# Extract class names. This is CRUCIAL for our web app later.
class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Found {num_classes} classes: {class_names}")

# Configure dataset for high performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(500).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# --- 3. Data Augmentation as a Model Layer ---
# This is a modern approach where augmentation happens on the GPU.
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
])

# --- 4. Build the Model with Transfer Learning ---
def build_model(num_classes):
    # Load Xception base, pre-trained on ImageNet, without the top classification layer
    base_model = tf.keras.applications.Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(args.img_size, args.img_size, 3)
    )
    # Freeze the base model so we only train our new layers
    base_model.trainable = False

    # Create our new model on top
    inputs = tf.keras.Input(shape=(args.img_size, args.img_size, 3))
    x = data_augmentation(inputs)  # Apply augmentation
    x = tf.keras.applications.xception.preprocess_input(x) # Preprocess input for Xception
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x) # Dropout for regularization
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model

model = build_model(num_classes)

# Compile the model with Adam optimizer and SparseCategoricalCrossentropy loss
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --- 5. Smart Training with Callbacks ---
# These callbacks make our training more efficient.
callbacks = [
    # Save only the best model based on validation accuracy
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(args.model_output_dir, 'leaf_disease_model.keras'),
        save_best_only=True,
        monitor='val_accuracy',
        verbose=1
    ),
    # Stop training early if there's no improvement
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5, # Stop if val_loss doesn't improve for 5 epochs
        verbose=1,
        restore_best_weights=True
    ),
    # Reduce learning rate if training plateaus
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2, # Reduce LR by a factor of 5
        patience=2,
        verbose=1
    )
]

# --- 6. Train the Model ---
print("\n🔥 Starting model training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=args.epochs,
    callbacks=callbacks
)

# --- 7. Save Final Assets ---
# Ensure the output directory exists
os.makedirs(args.model_output_dir, exist_ok=True)

# Save the class names to a JSON file. Our backend will use this.
class_names_path = os.path.join(args.model_output_dir, 'class_names.json')
with open(class_names_path, 'w') as f:
    json.dump(class_names, f)

print(f"\n✅ Training complete! Best model saved to the '{args.model_output_dir}' directory.")
print(f"Class names saved to '{class_names_path}'")

# --- 8. (Optional) Plot and Save Training History ---
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

# Save the plot
plot_path = os.path.join(args.model_output_dir, 'training_history.png')
plt.savefig(plot_path)
print(f"Training history plot saved to '{plot_path}'")
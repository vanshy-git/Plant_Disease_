import os
import subprocess
import tensorflow as tf
from tensorflow.keras import layers, models

# --- 1. SETUP LOCAL PATHS ---
# Get the current folder where this script is running
BASE_DIR = os.getcwd()
DATA_ROOT_DIR = os.path.join(BASE_DIR, 'dataset')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Create folders if they don't exist
os.makedirs(DATA_ROOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# --- 2. DOWNLOAD DATASET ---
# We use Python to run the git command locally
REPO_NAME = "PlantVillage-Dataset"
REPO_PATH = os.path.join(DATA_ROOT_DIR, REPO_NAME)
REPO_URL = "https://github.com/spMohanty/PlantVillage-Dataset"

if not os.path.exists(REPO_PATH):
    print(f"Downloading dataset to {DATA_ROOT_DIR}...")
    try:
        # This runs "git clone" in the dataset folder
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL], cwd=DATA_ROOT_DIR, check=True)
    except FileNotFoundError:
        print("❌ Error: Git is not installed or not in your PATH.")
        print("Please install Git or download the dataset manually.")
        exit()
else:
    print("Dataset already downloaded.")

# Point to the specific folder inside the repo
DATA_DIR = os.path.join(REPO_PATH, 'raw', 'color')

# --- 3. PREPARE DATA ---
BATCH_SIZE = 16 
IMG_SIZE = (224, 224)

print(f"Loading images from: {DATA_DIR}")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Save labels locally
class_names = train_dataset.class_names
labels_path = os.path.join(MODEL_DIR, 'labels.txt')

with open(labels_path, 'w') as f:
    for name in class_names:
        f.write(name + '\n')

# Optimize loading
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

# --- 4. BUILD MODEL ---
preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input
base_model = tf.keras.applications.MobileNetV3Small(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = preprocess_input(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(len(class_names), activation='softmax')(x)

model = models.Model(inputs, outputs)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

# --- 5. TRAIN ---
print("Starting Training on Local Machine...")
# Note: On a local CPU, this might take a while. 
# If you have a GPU setup, TensorFlow will try to use it.
model.fit(train_dataset, validation_data=validation_dataset, epochs=5)

# --- 6. SAVE MODEL ---
print("Converting and Saving model...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

tflite_path = os.path.join(MODEL_DIR, 'plant_disease_model.tflite')

with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

print(f"🎉 SUCCESS! Model saved to: {tflite_path}")
print(f"🎉 Labels saved to: {labels_path}")
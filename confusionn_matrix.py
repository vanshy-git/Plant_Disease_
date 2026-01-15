import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# --- SETTINGS ---
# CHANGE THIS TO WHERE YOUR DATASET IS! 
# If you ran train_local.py, it might be in a folder named 'dataset' next to this one.
DATASET_PATH = "dataset/PlantVillage-Dataset/raw/color" 
MODEL_PATH = "assets/plant_disease_model.tflite"
LABELS_PATH = "assets/labels.txt"

def load_brain():
    print("🔌 Connecting to the Brain...")
    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        
        with open(LABELS_PATH, 'r') as f:
            labels = [line.strip() for line in f.readlines()]
            
        return interpreter, labels
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None

def main():
    # 1. CHECK IF DATASET EXISTS
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Error: I cannot find the dataset at: {DATASET_PATH}")
        print("Please fix the DATASET_PATH line in the code!")
        return

    # 2. LOAD BRAIN
    interpreter, class_names = load_brain()
    if not interpreter: return
    
    input_idx = interpreter.get_input_details()[0]['index']
    output_idx = interpreter.get_output_details()[0]['index']

    print(f"📂 Found Dataset! Reading exam papers from: {DATASET_PATH}")
    print("🧠 Starting the Exam... (This will take time!)")

    y_true = []
    y_pred = []
    
    count = 0
    
    # 3. TAKE THE EXAM (Loop through folders)
    # We look at 30 images from each folder to save time (Total ~1000 images)
    # If you want to test ALL 54,000 images, remove the '[:30]' part.
    
    for class_idx, class_name in enumerate(class_names):
        folder_path = os.path.join(DATASET_PATH, class_name)
        if not os.path.exists(folder_path): continue
        
        images = os.listdir(folder_path)
        # ONLY TESTING 30 IMAGES PER CLASS TO BE FAST
        # Remove [:30] to test everything (might take 1 hour)
        images = images[:30] 
        
        print(f"   Testing {class_name} ({len(images)} images)...")
        
        for img_name in images:
            img_path = os.path.join(folder_path, img_name)
            
            # Prepare Image
            img = cv2.imread(img_path)
            if img is None: continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            input_data = np.expand_dims(img, axis=0).astype(np.float32)
            
            # Run Brain
            interpreter.set_tensor(input_idx, input_data)
            interpreter.invoke()
            output = interpreter.get_tensor(output_idx)
            
            # Record Answer
            predicted_idx = np.argmax(output[0])
            
            y_true.append(class_idx)
            y_pred.append(predicted_idx)
            count += 1

    print(f"✅ Exam Finished! Tested {count} images.")

    # 4. DRAW THE CHART
    print("🎨 Drawing the Report Card...")
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(15, 15))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    
    plt.ylabel('Actual Truth')
    plt.xlabel('Brain Predicted')
    plt.title('Confusion Matrix (TFLite Model)')
    plt.xticks(rotation=90)
    plt.tight_layout()

    output_file = "confusion_matrix.png"
    plt.savefig(output_file, dpi=300) 
    print(f"💾 Saved chart to {output_file}")

    plt.show()

if __name__ == "__main__":
    main()
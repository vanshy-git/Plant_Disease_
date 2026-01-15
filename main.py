import tensorflow as tf
import numpy as np
import cv2
import os
import requests
MODEL_PATH = "assets/plant_disease_model.tflite"
LABELS_PATH = "assets/labels.txt"
IMAGE_FOLDER = "test_images"
IMAGE_SIZE = 224

def load_brain():
    print(" Connecting to the Brain...")   
    
    if not os.path.exists(MODEL_PATH):
        print("Error: .tflite file missing in 'assets' folder.")
        return None, None
    
    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        
        with open(LABELS_PATH, 'r') as f:
            labels = [line.strip() for line in f.readlines()]
            
        print("Brain is ready!")
        return interpreter, labels
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def process_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not find image at {image_path}")
        return None

    cv2.imshow("Analyzing Leaf...", img)
    cv2.waitKey(500)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE))
    input_data = np.expand_dims(img_resized, axis=0).astype(np.float32)
    
    return input_data

def get_real_weather():
    print("\n Calling Weather Satellite (Moradabad, India)...")
    url = "https://api.open-meteo.com/v1/forecast?latitude=28.83&longitude=78.78&current=temperature_2m,relative_humidity_2m,rain"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        current = data['current']
        temp = current['temperature_2m']
        humidity = current['relative_humidity_2m']
        is_raining = current['rain'] > 0
        
        print(f"Temp: {temp}°C | Humidity: {humidity}% | Rain: {is_raining}")
        return temp, humidity, is_raining
    except:
        print("Internet Error! Using default data.")
        return 25.0, 50.0, False

def calculate_risk(visual_conf, temp, humidity, rain):
    weather_risk = 0.0
    
    if humidity > 85: weather_risk += 0.4
    elif humidity > 60: weather_risk += 0.2
        
    if 18 <= temp <= 26: weather_risk += 0.3
    if rain: weather_risk += 0.3
    
    weather_risk = min(weather_risk, 1.0)
    final_score = (visual_conf * 0.7) + (weather_risk * 0.3)
    return final_score, weather_risk

def main():
    interpreter, labels = load_brain()
    if not interpreter: return

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    while True:
        print("\n" + "="*50)
        image_name = input("Enter image name (e.g., leaf.jpg) or 'q' to quit: ")
        if image_name.lower() == 'q': break
        
        image_path = os.path.join(IMAGE_FOLDER, image_name)
        input_data = process_image(image_path)
        
        if input_data is None: continue
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediction = np.squeeze(output_data)
        
        winner_index = np.argmax(prediction)
        disease_name = labels[winner_index]
        visual_conf = prediction[winner_index]
        temp, humidity, rain = get_real_weather()
        final_risk, weather_risk = calculate_risk(visual_conf, temp, humidity, rain)
        print("\n" + "*"*40)
        print(f" VISUAL: {disease_name} ({visual_conf*100:.1f}%)")
        print(f" WEATHER RISK: {weather_risk*100:.1f}%")
        print("-" * 40)
        print(f" TOTAL RISK SCORE: {final_risk*100:.1f}%")
        print("*"*40)
        
        if final_risk >= 0.90:
            print("EXTREME RISK")
            print("Conditions strongly support rapid disease development. Immediate spraying is required to prevent major crop damage.")

        elif final_risk >= 0.75:
            print("VERY HIGH RISK")
            print("Disease pressure is significantly high due to environmental conditions. Spraying should be done as soon as possible.")

        elif final_risk >= 0.60:
            print("HIGH RISK")
            print("Weather conditions are becoming favorable for disease growth. Early infection may already be starting. Prepare treatment and inspect the field within 12 hours.")

        elif final_risk >= 0.40:
            print("MODERATE RISK")
            print("There are some indicators of potential infection. Spraying is not required yet, but close monitoring is necessary.")

        elif final_risk >= 0.25:
            print("LOW TO MODERATE RISK")
            print("Disease pressure is currently low but could increase with minor weather changes. Light monitoring is recommended.")

        elif final_risk >= 0.10:
            print("LOW RISK")
            print("Environmental conditions are generally unfavorable for disease. No immediate action is needed.")

        else:
            print("VERY LOW RISK")
            print("Conditions do not support disease development. The field is safe and no intervention is required.")


            
        print("\n(Press any key on image window to continue...)")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
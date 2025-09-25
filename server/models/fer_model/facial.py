import tensorflow as tf
import numpy as np
import cv2
import base64
import io
from PIL import Image
from collections import Counter

# Define the expected emotion labels for your model.
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
# Define the input image size your model was trained on.
IMG_SIZE = (48, 48)
def load_model(model_path):
    """
    Loads the Keras model from the specified path.
    """
    print(f"[*] Loading facial emotion model from: {model_path}")
    try:
        model = tf.keras.models.load_model(model_path)
        print("[+] Facial emotion model loaded successfully.")
        return model
    except Exception as e:
        print(f"[!] Error loading facial emotion model: {e}")
        return None

def visual_predict(model, image_data_base64):
    """
    Takes a base64 encoded image string, preprocesses it, 
    and returns the predicted emotion label.
    """
    if not image_data_base64:
        return "N/A"

    try:
        # 1. Decode the Base64 string
        if ',' in image_data_base64:
            _, encoded = image_data_base64.split(',', 1)
        else:
            encoded = image_data_base64
        
        image_bytes = base64.b64decode(encoded)
        
        # 2. Convert bytes to an OpenCV image
        image = Image.open(io.BytesIO(image_bytes))
        image = np.array(image)
        if len(image.shape) > 2 and image.shape[2] == 4:
            # Drop the alpha channel
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        # 3. Preprocess the image for the model
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(gray_image, IMG_SIZE)
        normalized_image = resized_image / 255.0
        reshaped_image = np.expand_dims(normalized_image, axis=-1)   # (48,48,1)
        reshaped_image = np.expand_dims(reshaped_image, axis=0)      # (1,48,48,1)

        # 4. Run prediction
        prediction = model.predict(reshaped_image, verbose=0)
        
        # 5. Get the emotion label
        max_index = np.argmax(prediction[0])
        V_predicted_emotion = EMOTION_LABELS[max_index]
        
        return V_predicted_emotion

    except Exception as e:
        print(f"[!] Error during facial emotion prediction: {e}")
        return "Error"

def video_predict(model, video_base64_frames):
    """
    Takes a list of base64 frames from frontend video,
    classifies each frame, and returns the majority emotion.
    """
    predictions = []

    for frame_base64 in video_base64_frames:
        emotion = visual_predict(model, frame_base64)  # reuse image classifier
        if emotion not in ["N/A", "Error"]:
            predictions.append(emotion)

    if not predictions:
        return "N/A"

    # Majority vote
    most_common = Counter(predictions).most_common(1)[0][0]
    return most_common


def video_to_base64_frames(video_path, max_frames=50):
    """
    Reads a video file and converts frames to base64 strings.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[!] Error: Could not open video file:", video_path)
        return []
    frames = []
    count = 0
    
    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)

        # Convert to base64 string
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        frames.append(frame_base64)

        count += 1

    cap.release()
    return frames


def video_predict_rag(model, video_frames):
    V_predicted_emotion = video_predict(model, video_frames)  # original function
    return {
        "source": "video",
        "type": "emotion",
        "content": f"The person’s facial expression is {V_predicted_emotion}.",
        "raw_label": V_predicted_emotion
    }

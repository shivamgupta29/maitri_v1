import tensorflow as tf
import numpy as np
import librosa
import base64
import io
import os

# Define the expected emotion labels for your model.
# The order MUST match the output order of your trained model.
EMOTION_LABELS = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
# Define audio processing parameters
SAMPLE_RATE = 22050
MAX_PAD_LENGTH = 174 # Example value, adjust to your model's needs

def load_model(model_path):
    """
    Loads the Keras model for speech emotion recognition.
    """
    print(f"[*] Loading speech emotion model from: {model_path}")
    try:
        model = tf.keras.models.load_model(f"{model_path}")
        print("[+] Speech emotion model loaded successfully.")
        return model
    except Exception as e:
        print(f"[!] Error loading speech emotion model: {e}")
        return None

def speech_predict(model, audio_data_base64):
    """
    Takes a base64 encoded audio string, preprocesses it, and returns the predicted emotion.
    """
    if not audio_data_base64:
        return "N/A"
        
    try:
        # 1. Decode the Base64 string
        if ',' in audio_data_base64:
            header, encoded = audio_data_base64.split(',', 1)
        else:
            encoded = audio_data_base64

        audio_bytes = base64.b64decode(encoded)
        audio_file = io.BytesIO(audio_bytes)

        # 2. Load audio and extract features using Librosa
        audio, sample_rate = librosa.load(audio_file, sr=SAMPLE_RATE, res_type='kaiser_fast') 
        
        # Extract Mel-Frequency Cepstral Coefficients (MFCCs)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        
        # Pad or truncate the features to a fixed length
        if mfccs.shape[1] > MAX_PAD_LENGTH:
            mfccs = mfccs[:, :MAX_PAD_LENGTH]
        else:
            pad_width = MAX_PAD_LENGTH - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')

        # 3. Reshape for model prediction
        # The model likely expects (1, num_mfcc, max_length, 1)
        reshaped_features = np.expand_dims(mfccs, axis=-1)
        reshaped_features = np.expand_dims(reshaped_features, axis=0)

        # 4. Run prediction
        prediction = model.predict(reshaped_features)
        
        # 5. Get the emotion label
        max_index = np.argmax(prediction[0])
        S_predicted_emotion = EMOTION_LABELS[max_index]

        return S_predicted_emotion

    except Exception as e:
        print(f"[!] Error during speech emotion prediction: {e}")
        return "Error"
    

def speech_predict_rag(model, audio_base64):
    S_predicted_emotion = speech_predict(model, audio_base64)  # original function
    return {
        "source": "speech",
        "type": "emotion",
        "content": f"The speaker’s tone is {S_predicted_emotion}.",
        "raw_label": S_predicted_emotion
    }
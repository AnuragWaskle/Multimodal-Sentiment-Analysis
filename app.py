import streamlit as st
import nltk
import io
import tempfile
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import speech_recognition as sr
from pydub import AudioSegment
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
from nltk.tokenize import word_tokenize
import string

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('wordnet')

# Sentiment analysis function
def sentiment_analyse(sentiment_text):
    score = SentimentIntensityAnalyzer().polarity_scores(sentiment_text)
    neg = score['neg']
    pos = score['pos']
    if neg > pos:
        return "Negative sentiment"
    elif pos > neg:
        return "Positive sentiment"
    else:
        return "Neutral sentiment"

# Load emotions from file
emotions = {}
try:
    with open('emotions.txt', 'r') as file:
        for line in file:
            clear_line = line.replace("\n", '').replace(",", '').replace("'", '').strip()
            word, emotion = clear_line.split(':')
            emotions[word] = emotion.capitalize()
except FileNotFoundError:
    st.error("emotions.txt file not found!")

# Emotion dictionary for face detection
emotion_dict = {0: 'angry', 1: 'fear', 2: 'happy', 3: 'neutral', 4: 'sad', 5: 'surprise'}

# Load face emotion model
try:
    with open('face_emotion.json', 'r') as json_file:
        loaded_model_json = json_file.read()
    classifier = model_from_json(loaded_model_json)
    classifier.load_weights("face_emotion.h5")
    st.write("Model loaded successfully")
    # Debug model input shape
    st.write(f"Model expected input shape: {classifier.input_shape}")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")

# Load face cascade classifier
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except Exception as e:
    st.error(f"Error loading cascade classifiers: {str(e)}")

# Audio recording function
def record_audio():
    recognizer = sr.Recognizer()
    text = ""
    try:
        with sr.Microphone(device_index=1) as source:
            st.write('Clearing background noise...')
            recognizer.adjust_for_ambient_noise(source, duration=3)
            st.write('Start Speaking...')
            start_time = time.time()
            recorded_audio = recognizer.listen(source)
            end_time = time.time()
            st.write(f'Done recording! Time taken: {round(end_time - start_time, 2)} seconds')
            text = recognizer.recognize_google(recorded_audio, language='en-US')
            st.write(f'Your message: {text}')
    except Exception as ex:
        st.error(f"Audio recording error: {str(ex)}")
    return text

# Audio file recognition
def recognize_audio(uploaded_audio):
    recognizer = sr.Recognizer()
    text = ""
    try:
        # Check if ffmpeg is available
        import subprocess
        subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        audio_file = io.BytesIO(uploaded_audio.read())
        try:
            audio = AudioSegment.from_file(audio_file, format="wav")
        except Exception as format_error:
            st.error(f"Audio format error: {str(format_error)}. Ensure the file is a valid WAV file.")
            return text
            
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
            audio.export(temp_audio_file.name, format="wav")
        
        with sr.AudioFile(temp_audio_file.name) as source:
            st.write('Clearing background noise...')
            recognizer.adjust_for_ambient_noise(source, duration=3)
            st.write('Analysing uploaded wav file...')
            recorded_audio = recognizer.record(source)
            duration_seconds = len(audio) / 1000
            st.write(f'Done! Duration: {duration_seconds} seconds')
            text = recognizer.recognize_google(recorded_audio, language='en-US')
            st.write(f'Your message: {text}')
    except subprocess.CalledProcessError:
        st.error("ffmpeg is not installed or not found in PATH. Please install ffmpeg and add it to your system PATH.")
    except Exception as ex:
        st.error(f"Audio processing error: {str(ex)}")
    return text

# Video transformer for live sentiment analysis
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame_count = 0

    def transform(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(image=img_gray, scaleFactor=1.3, minNeighbors=5)
            
            self.frame_count += 1
            print(f"Frame {self.frame_count}: Detected {len(faces)} faces")  # Debug
            
            # Display "No face detected" if no faces are found
            if len(faces) == 0:
                cv2.putText(img, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0, 0, 255), 2, cv2.LINE_AA)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(img=img, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
                face_image = img_gray[y:y + h, x:x + w]
                if face_image.size == 0:
                    print("Empty face region detected")
                    continue
                face_image = cv2.resize(face_image, (48, 48), interpolation=cv2.INTER_AREA)
                face_image = face_image.astype('float32') / 255.0
                face_image = img_to_array(face_image)
                face_image = np.expand_dims(face_image, axis=0)  # Shape: (1, 48, 48, 1)
                
                try:
                    prediction = classifier.predict(face_image)[0]
                    max_index = int(np.argmax(prediction))
                    predicted_emotion = emotion_dict.get(max_index, "Unknown")
                    print(f"Prediction: {prediction}, Emotion: {predicted_emotion}")  # Debug
                    
                    # Display emotion live on the frame
                    cv2.rectangle(img, (x, y - 40), (x + w, y), (0, 0, 255), -1)
                    cv2.putText(img, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.9, (255, 255, 255), 2, cv2.LINE_AA)
                except Exception as e:
                    print(f"Prediction error: {str(e)}")
                    cv2.putText(img, "Prediction failed", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.9, (255, 255, 255), 2, cv2.LINE_AA)
            
            return img
        except Exception as e:
            print(f"Video transform error: {str(e)}")
            return frame  # Return original frame if processing fails

# Main Streamlit app
def main():
    st.title("Sentiment Analysis App")
    analysis_type = st.sidebar.selectbox("Choose Analysis Type", ["Text Analysis", "Image Analysis", "Audio Analysis", "Video Analysis"])

    if analysis_type == "Text Analysis":
        st.write("Enter a text and get the predicted Sentiment & Emotions")
        user_input = st.text_area("Text", height=100)
        if st.button("Predict") and user_input:
            cleaned_text = user_input.lower().translate(str.maketrans('', '', string.punctuation))
            tokenized_words = word_tokenize(cleaned_text, "english")
            final_words = [word for word in tokenized_words if word not in stopwords.words('english')]
            detected_emotions_set = {emotions[word] for word in final_words if word in emotions}
            detected_emotions = sorted(detected_emotions_set)
            sentiment_result = sentiment_analyse(cleaned_text)
            st.success(f"Detected Sentiment: {sentiment_result}")
            st.success("Detected Emotions: " + ", ".join(emotion.title() for emotion in detected_emotions)) if detected_emotions else st.info("No emotions detected.")

    elif analysis_type == "Image Analysis":
        st.write("Upload an Image and get the predicted Emotion")
        uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])
        
        if uploaded_file:
            pil_image = Image.open(uploaded_file)
            original_image = np.array(pil_image)
            st.image(original_image, use_container_width=True)
            
            if st.button("Predict"):
                image_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY) if original_image.ndim == 3 else original_image
                faces = face_cascade.detectMultiScale(image=image_gray, scaleFactor=1.3, minNeighbors=5)
                
                if len(faces) == 0:
                    st.error("No human face detected.")
                else:
                    try:
                        for (x, y, w, h) in faces:
                            cv2.rectangle(img=original_image, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
                            face_image = image_gray[y:y+h, x:x+w]
                            if face_image.size == 0:
                                st.warning("Empty face region detected")
                                continue
                            face_image = cv2.resize(face_image, (48, 48), interpolation=cv2.INTER_AREA)
                            face_image = face_image.astype('float32') / 255.0
                            face_image = img_to_array(face_image)
                            face_image = np.expand_dims(face_image, axis=0)  # Shape: (1, 48, 48, 1)
                            
                            prediction = classifier.predict(face_image)[0]
                            max_index = int(np.argmax(prediction))
                            predicted_emotion = emotion_dict.get(max_index, "Unknown")
                            st.write(f"Prediction: {prediction}, Emotion: {predicted_emotion}")  # Debug output
                            
                            cv2.rectangle(original_image, (x, y - 40), (x + w, y), (0, 0, 255), -1)
                            cv2.putText(original_image, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                       0.9, (255, 255, 255), 2, cv2.LINE_AA)
                        
                        st.image(original_image, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error in face processing: {str(e)}")

    elif analysis_type == "Audio Analysis":
        st.header("Audio Sentiments Analysis")
        audio_option = st.selectbox("Choose Audio Option", ["Upload Audio", "Record Audio"])
        
        if audio_option == "Upload Audio":
            uploaded_audio = st.file_uploader("Upload an audio file", type=["wav"])
            if uploaded_audio:
                st.audio(uploaded_audio, format="audio/wav")
                if st.button("Predict"):
                    audio_text = recognize_audio(uploaded_audio)
                    if audio_text:
                        cleaned_text = audio_text.lower().translate(str.maketrans('', '', string.punctuation))
                        tokenized_words = word_tokenize(cleaned_text, "english")
                        final_words = [word for word in tokenized_words if word not in stopwords.words('english')]
                        detected_emotions_set = {emotions[word] for word in final_words if word in emotions}
                        detected_emotions = sorted(detected_emotions_set)
                        sentiment_result = sentiment_analyse(cleaned_text)
                        st.success(f"Detected Sentiment: {sentiment_result}")
                        st.success("Detected Emotions: " + ", ".join(emotion.title() for emotion in detected_emotions)) if detected_emotions else st.info("No emotions detected.")
                    else:
                        st.warning("No text extracted from audio. Ensure ffmpeg is installed and the audio file is valid.")
        
        elif audio_option == "Record Audio":
            if st.button("Start Recording"):
                audio_text = record_audio()
                if audio_text:
                    cleaned_text = audio_text.lower().translate(str.maketrans('', '', string.punctuation))
                    tokenized_words = word_tokenize(cleaned_text, "english")
                    final_words = [word for word in tokenized_words if word not in stopwords.words('english')]
                    detected_emotions_set = {emotions[word] for word in final_words if word in emotions}
                    detected_emotions = sorted(detected_emotions_set)
                    sentiment_result = sentiment_analyse(cleaned_text)
                    st.success(f"Detected Sentiment: {sentiment_result}")
                    st.success("Detected Emotions: " + ", ".join(emotion.title() for emotion in detected_emotions)) if detected_emotions else st.info("No emotions detected.")
                else:
                    st.warning("No text recorded. Check your microphone and permissions.")

    elif analysis_type == "Video Analysis":
        st.header("Live Video Sentiment Analysis")
        st.write("Click 'Start' to begin live emotion detection from your webcam.")
        # Use RTCConfiguration for stable WebRTC connection
        rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
        try:
            webrtc_streamer(
                key="live-sentiment",
                video_processor_factory=VideoTransformer,
                rtc_configuration=rtc_config,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True  # Enable asynchronous processing for smoother performance
            )
            st.info("Emotion detection updates in real-time as faces are detected.")
        except Exception as e:
            st.error(f"Error initializing live video feed: {str(e)}")
            st.write("Ensure your webcam is connected and browser permissions are granted.")

if __name__ == '__main__':
    main()
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import webbrowser
import time

# Load the emotion detection model
def custom_load_model(filepath):
    return load_model(filepath, compile=False)

emotion_model = custom_load_model('emotion_model.hdf5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Set up Spotify client
client_id = '478d01654f9c42d2a029e80ffd3ba09c'  # Replace with your Client ID
client_secret = '467a0e998a53481a818af2c49794a8e7'  # Replace with your Client Secret

spotify = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))

# Map emotions to playlist search terms
emotion_to_playlist = {
    'Happy': 'happy songs',
    'Sad': 'sad songs',
    'Angry': 'angry music',
    'Neutral': 'chill music',
}

# Function to fetch and open Spotify playlist
def get_spotify_playlist(emotion):
    if emotion in emotion_to_playlist:
        search_query = emotion_to_playlist[emotion]
        results = spotify.search(q=search_query, type='playlist', limit=1)
        if results['playlists']['items']:
            playlist_url = results['playlists']['items'][0]['external_urls']['spotify']
            print(f"Opening playlist for {emotion}: {playlist_url}")
            webbrowser.open(playlist_url)
        else:
            print(f"No playlist found for emotion: {emotion}")

# Initialize webcam
cap = cv2.VideoCapture(0)

# Start the timer for 15-20 seconds
start_time = time.time()

# Initialize the emotion variable
detected_emotion = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Process faces in the video feed
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (64, 64))
        roi_gray = roi_gray.astype("float") / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=-1)
        roi_gray = np.expand_dims(roi_gray, axis=0)

        # Predict emotion
        emotion_prediction = emotion_model.predict(roi_gray)
        max_index = int(np.argmax(emotion_prediction))
        emotion = emotion_labels[max_index]

        # Display the emotion label on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Store the detected emotion after the timer has passed
        if time.time() - start_time > 15 and detected_emotion is None:
            detected_emotion = emotion  # Set the detected emotion after 15 seconds
            print(f"Emotion detected after 15 seconds: {emotion}")
            get_spotify_playlist(emotion)
            break

    # Show the video feed
    cv2.imshow('Emotion Recognition', frame)

    # Check if 15-20 seconds have passed to decide the emotion and suggest a playlist
    if detected_emotion is not None:
        break

    # Stop the program if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

from flask import Flask, render_template_string, Response, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os


app = Flask(__name__)


emotion_model = load_model('emotion_model.hdf5', compile=False)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID', '478d01654f9c42d2a029e80ffd3ba09c')
CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET','467a0e998a53481a818af2c49794a8e7')

spotify = spotipy.Spotify(
    auth_manager=SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
)


emotion_to_playlist = {
    'Happy': 'happy songs',
    'Sad': 'sad songs',
    'Angry': 'angry music',
    'Neutral': 'chill music',
}


def initialize_video_capture():
    """Initialize video capture."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Error: Could not open video capture.")
    return cap

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/')
def index():
    """Serve the main HTML page."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Emotion Detection</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; background: #f4f4f4; margin: 0; padding: 20px; }
            h1 { color: #4CAF50; }
            img { border: 1px solid #ddd; border-radius: 4px; padding: 5px; width: 50%; }
            button { margin-top: 20px; padding: 10px 20px; background-color: #4CAF50; color: white; border: none; border-radius: 5px; cursor: pointer; }
            button:hover { background-color: #45a049; }
        </style>
    </head>
    <body>
        <h1>Live Emotion Detection</h1>
        <p>Emotion detection and music recommendation are running in real-time.</p>
        <div>
            <img src="/video_feed" alt="Live Video Feed">
        </div>
        <button onclick="detectEmotion()">Detect Emotion</button>
        <p id="emotion"></p>
        <p id="playlist"></p>
        <script>
            function detectEmotion() {
                fetch('/detect', { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('emotion').innerText = `Emotion: ${data.emotion}`;
                        if (data.playlist) {
                            document.getElementById('playlist').innerHTML = `<a href="${data.playlist}" target="_blank">Open Playlist</a>`;
                        } else {
                            document.getElementById('playlist').innerText = "No playlist found.";
                        }
                    })
                    .catch(error => {
                        console.error("Error:", error);
                        document.getElementById('playlist').innerText = "Error fetching playlist.";
                    });
            }
        </script>
    </body>
    </html>
    """
    return render_template_string(html_content)

@app.route('/video_feed')
def video_feed():
    """Serve the live video feed."""
    def generate_frames():
        cap = initialize_video_capture()
        while True:
            success, frame = cap.read()
            if not success:
                print("Error: Could not read frame.")
                break
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray)
                print("Detected faces:", faces)

                for (x, y, w, h) in faces:
                    roi_gray = gray[y:y + h, x:x + w]
                    roi_gray = cv2.resize(roi_gray, (64, 64))
                    roi_gray = roi_gray.astype('float32') / 255.0
                    roi_gray = np.expand_dims(roi_gray, axis=-1)
                    roi_gray = np.expand_dims(roi_gray, axis=0)

                    emotion_prediction = emotion_model.predict(roi_gray, verbose=0)
                    print("Emotion Prediction:", emotion_prediction)
                    max_index = np.argmax(emotion_prediction)
                    detected_emotion = emotion_labels[max_index]
                    print("Detected Emotion:", detected_emotion)

                    
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(frame, detected_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

               
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    print("Error: Could not encode frame.")
                    continue
                frame = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        cap.release() 

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect', methods=['POST'])
def detect_emotion():
    """Handle emotion detection and recommend a playlist."""
    try:
        detected_emotion = None
        cap = initialize_video_capture()
        start_time = time.time()

        
        while time.time() - start_time < 5:
            success, frame = cap.read()
            if not success:
                print("Error: Could not read frame during detection.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray)
            print("Faces:", faces)

            for (x, y, w, h) in faces:
                roi_gray = cv2.resize(gray[y:y + h, x:x + w], (64, 64)) / 255.0
                roi_gray = np.expand_dims(roi_gray, axis=(0, -1))

                emotion_prediction = emotion_model.predict(roi_gray)
                print("Emotion Prediction:", emotion_prediction)
                detected_emotion = emotion_labels[np.argmax(emotion_prediction)]
                print("Detected Emotion:", detected_emotion)

                if detected_emotion:  
                    break
            if detected_emotion:
                break

        cap.release()  

        if detected_emotion and detected_emotion in emotion_to_playlist:
            search_query = emotion_to_playlist[detected_emotion]
            results = spotify.search(q=search_query, type='playlist', limit=1)
            print("Spotify search results:", results)

            if results['playlists']['items']:
                playlist_url = results['playlists']['items'][0]['external_urls']['spotify']
                return jsonify({'emotion': detected_emotion, 'playlist': playlist_url})
            else:
                return jsonify({'emotion': detected_emotion, 'playlist': None, 'message': 'No playlist found.'})
        else:
            return jsonify({'emotion': detected_emotion, 'playlist': None, 'message': 'No matching playlist for emotion.'})
    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

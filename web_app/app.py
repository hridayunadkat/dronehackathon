from flask import Flask, render_template, request, jsonify
import speech_recognition as sr
import os
from datetime import datetime

app = Flask(__name__)

# Global variables
recognizer = sr.Recognizer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    # try:   
    #     # Initialize recognizer
    #     r = sr.Recognizer()

    #     # Use the default microphone as the audio source
    #     with sr.Microphone() as source:
    #         print("Please say something...")
    #         # Adjust for ambient noise for 1 second
    #         r.adjust_for_ambient_noise(source, duration=1)
    #         # Listen for the first phrase and extract it into audio data
    #         audio = r.listen(source)
    #         print("Recognizing...")

    try:
        # Recognize speech using Google Speech Recognition
        text = sr.recognize_google(audio)
        print("You said: " + text)
    except sr.UnknownValueError:
        print("Sorry, I could not understand the audio.")
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))
        audio_file = request.files['audio']
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            return jsonify({'text': text})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True) 
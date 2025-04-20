import speech_recognition as sr

# Initialize recognizer
r = sr.Recognizer()

# Use the default microphone as the audio source
with sr.Microphone() as source:
    print("Please say something...")
    # Adjust for ambient noise for 1 second
    r.adjust_for_ambient_noise(source, duration=1)
    # Listen for the first phrase and extract it into audio data
    audio = r.listen(source)
    print("Recognizing...")

    try:
        # Recognize speech using Google Speech Recognition
        text = r.recognize_google(audio)
        print("You said: " + text)
    except sr.UnknownValueError:
        print("Sorry, I could not understand the audio.")
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))

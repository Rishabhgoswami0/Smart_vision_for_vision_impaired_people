import speech_recognition as sr
import text_to_speech
import pyttsx3

engine = pyttsx3.init()


def recog_command():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = r.listen(source, phrase_time_limit=5)  # Adjust the phrase_time_limit as needed
        
    try:
        print("Recognizing...")
        command = r.recognize_google(audio, language="en-US")
        command = command.lower()
        print("You said: " + command)
        text_to_speech.speak("You said: " + command)
        return command
    except sr.UnknownValueError:
        print("Sorry, I didn't understand that.")
        text_to_speech.speak("Sorry, I didn't understand that.")
        return ""
    except sr.RequestError as e:
        print("Sorry, I couldn't request results from Google Speech Recognition service; {0}".format(e))
        text_to_speech.speak("Sorry, I couldn't request results from Google Speech Recognition service; {0}".format(e))
        return ""

import speech_recognition as sr
import text_to_speech
import subprocess
import time
# Global flag variable
subprocess_running = False
capture_audio = True
exit_flag = False
# Function to handle the recognized commands
def handle_command(command):
    global subprocess_running , capture_audio,exit_flag
    if "face recognition" in command:
        subprocess_running = True        
        text_to_speech.speak("You command for face recognition")
        face_detection_cmd = ["python", "/home/abvv/Major_Project/facial_recognition/facial_recog.py"]
        # Run the face detection program in a subprocess
        subprocess.run(face_detection_cmd)
        subprocess_running = False
       
    elif "text reading" in command:
        subprocess_running = True
        text_to_speech.speak("You command for text reading")
        text_detection_cmd = [
            "python",
            "/home/abvv/Major_Project/text_recognition/text_detect.py",
            "-i", "/home/abvv/Major_Project/text_recognition/captured_image.jpg",
            "-east", "/home/abvv/Major_Project/text_recognition/frozen_east_text_detection.pb",
            "-c", "0.7",
            "-w", "640",
            "-e", "320"]
        # Run the text detection program in a subprocess
        subprocess.run(text_detection_cmd)
        subprocess_running = False
    elif "exit" in command:
        exit_flag = True
        text_to_speech.speak("Exiting the voice assistant.")   

# Function to listen for the "hello" keyword and commands
def listen():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=2)
   # Check if subprocess is running
        if subprocess_running:
            text_to_speech.speak("Please wait for the current process to complete.")
            capture_audio = False
        else:
            capture_audio = True
        
        print("Listening...")
        if capture_audio:
            audio = recognizer.listen(source, phrase_time_limit=2)
        else:
            time.sleep(0.8)

        try:
            print("Recognizing...")
            recognized_text = recognizer.recognize_google(audio)
            print("Recognized:", recognized_text)
            
            if "hello" in recognized_text.lower():
                text_to_speech.speak("say 'face recognition' or 'text reading' or say 'exit' to quit.")
                print("Listening...")
                recognizer.adjust_for_ambient_noise(source)
                retries = 3
                for _ in range(retries):
                    try:
                        if subprocess_running:
                            text_to_speech.speak("Please wait for the current process to complete.")
                            break
                        audio = recognizer.listen(source)
                        command = recognizer.recognize_google(audio)
                        command = command.lower()
                        print("You said:", command)
                        handle_command(command)
                        
                    except sr.UnknownValueError:
                        pass

                    except sr.RequestError as e:
                        print("Error occurred during recognition: {0}. Retrying...".format(e))
                        text_to_speech.speak("Error occurred during recognition. Please try again.")
            else:
                print("No valid command detected.")
                text_to_speech.speak("No valid command detected. Please try again.")

        except sr.UnknownValueError:
            text_to_speech.speak("Could not understand audio.")
        except sr.RequestError as e:
            text_to_speech.speak("Error occurred during recognition: {0}".format(e))
         
# Start the voice assistant
if __name__ == "__main__":
    text_to_speech.speak("Say 'hello' to activate the voice assistant.")
    while True:
        if not subprocess_running:
            listen()
        else:
            time.sleep(5)
        
        if exit_flag:
            break

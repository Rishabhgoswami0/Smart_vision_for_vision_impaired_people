import os
# Function to convert text to speech
def speak(string):
    text = string

# create a temporary file to store the text
    with open('temp.txt', 'w') as f:
        f.write(text)

# call Festival to convert the text to speech
    os.system('festival --tts temp.txt')

# remove the temporary file
    os.remove('temp.txt')

#speak("hello  good")
import text_to_speech
import os
# it takes the detected text from text_detect.py
# and writes and speak the words
def writing(detected_text):
	with open("extracted_text.txt", "w") as file:
		pass
	# Read the existing lines from the file
	with open("extracted_text.txt", "r") as file:
		existing_lines = file.readlines()
	# Clear the file before writing new session lines
	with open("extracted_text.txt", "w") as file:
		for line in existing_lines[:-1]:
			file.write(line)
		# Append a space after the last line
		if existing_lines:
			file.write(existing_lines[-1].rstrip() + " ")  

		for (bbox, text, prob) in detected_text:
			# Add space after each word
			text_with_space = text + " "
			file.write(text_with_space)
			print(f"Text: {text}\n")
	# Read the existing lines from the file
	with open("extracted_text.txt", "r") as file:
		text_lines = file.readlines()
		# Convert the list of lines into a single string
		text = ''.join(text_lines)
		text_to_speech.speak(text)

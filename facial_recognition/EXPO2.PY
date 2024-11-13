# import the necessary packages
from imutils.video import VideoStream, FPS
import face_recognition
import imutils
import pickle
import cv2
import text_to_speech
import train_model
import speech_to_text
import time
import os
 

#Initialize 'currentname' to trigger only when a new person is identified.
currentname = "unknown"
#Determine faces from encodings.pickle file model created from train_model.py
encodingsP = "encodings.pickle"
# load the known faces and embeddings along with OpenCV's Haar
# cascade for face detection
print("[INFO] loading encodings + face detector...")
data = pickle.loads(open(encodingsP, "rb").read())

# initialize the video stream and allow the camera sensor to warm up
# Set the ser to the followng
# src = 0 : for the build in single web cam, could be your laptop webcam
# src = 2 : I had to set it to 2 inorder to use the USB webcam attached to my laptop
vs = VideoStream(src=0).start()
#vs = VideoStream(usePiCamera=True).start()

# start the FPS counter
fps = FPS().start()

# this function takes input from user in given time interval

def get_user_input(timeout):
    start_time = time.time()
    elapsed_time = 0

    while elapsed_time < timeout:
        command = speech_to_text.recog_command()#uses voice recognition import from speech_to_text.py 
        if command != "":
            return command

        elapsed_time = time.time() - start_time
    text_to_speech.speak("You didn't respond in given Time, try again!!")
    return None

# ...

def camera_call():
    text_to_speech.speak("Initializing Data Collection")
    text_to_speech.speak("say person's name")
    print("Say the Person's Name")
    while True:
        folder_name = get_user_input(timeout=10)
        if folder_name and folder_name.strip()!= "":
            parent_path = "dataset/"
            new_folder_path = os.path.join(parent_path, folder_name)
            break
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
        name = folder_name
        img_counter = 0
        for i in range (20):
            frame = vs.read()
            if frame is None:
                print("Failed to grab frame")
                break
            frame = imutils.resize(frame, width=360)
            cv2.imshow("frame", frame)
            img_name = "dataset/{}/image_{}.jpg".format(name, img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1
            cv2.waitKey(1000)

        cv2.destroyAllWindows()
        train_model.training()

# loop over frames from the video file stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to 500px (to speedup processing)
	frame = vs.read()
	frame = imutils.resize(frame, width=360)
	# Detect the fce boxes
	boxes = face_recognition.face_locations(frame)
	# compute the facial embeddings for each face bounding box
	encodings = face_recognition.face_encodings(frame, boxes)
	names = []

	# loop over the facial embeddings
	for encoding in encodings:
		# attempt to match each face in the input image to our known
		# encodings
		matches = face_recognition.compare_faces(data["encodings"],encoding)
		global name #use golbal to use all over the code
		name = 'unknown'
		# check to see if we have found a match
		if True in matches:
			# find the indexes of all matched faces then initialize a
			# dictionary to count the total number of times each face
			# was matched
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}

			# loop over the matched indexes and maintain a count for
			# each recognized face face
			for i in matchedIdxs:
				name = data["names"][i]
				counts[name] = counts.get(name, 0) + 1

			# determine the recognized face with the largest number
			# of votes (note: in the event of an unlikely tie Python
			# will select first entry in the dictionary)
			name = max(counts, key=counts.get)
		# update the list of names
		names.append(name)
                               

	# loop over the recognized faces
	for ((top, right, bottom, left), name) in zip(boxes, names):
		# draw the predicted face name on the image - color is in BGR
		cv2.rectangle(frame, (left, top), (right, bottom),
			(255, 255, 225), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			.8, (0, 255, 255), 2)
		if currentname != name: #If someone in your dataset is identified, print their name on the screen
			currentname = name
			print(currentname)
			text_to_speech.speak(currentname)
		if name.lower().strip() == "unknown":
			currentname = name
			print("An Unknown Person is Detected")
			text_to_speech.speak("An Unknown Person is Detected")
			text_to_speech.speak("Do you want to save it, YES or NO")
			command = None
			command = get_user_input(timeout=10)
			if command is not None:
				if command.lower().strip() =='yes':
					camera_call()
				elif command.lower().strip() =='no':
					pass

	# display the image to our screen
	cv2.imshow("Facial Recognition is Running", frame)
	
	if cv2.waitKey(10) == ord('q'):
		break  # exit loop if 'q' k

	# update the FPS counter
	fps.update()
# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup

cv2.destroyAllWindows()


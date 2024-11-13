import cv2
import imutils
import matplotlib.pyplot as plt
import time
#function to capture image through cv2
def cam():
	# Capture a frame from the camera
	cap = cv2.VideoCapture(0)
	ret, frame = cap.read()

	if not ret:
		print("Failed to grab frame")

	
	time.sleep(0.8)

	# Capture a good image
	image_path = "/home/abvv/Major_Project/text_recognition/captured_image.jpg"
	cv2.imwrite(image_path, frame)
	print("Image saved:", image_path)

	# Read the captured image
	img = cv2.imread(image_path)
	image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	# Show the output image
# 	plt.imshow(image_rgb)
# 	plt.axis("off")
	# plt.pause(3)
	cv2.waitKey(0)

	# Release the camera and close windows
	cap.release()
	cv2.destroyAllWindows()
#
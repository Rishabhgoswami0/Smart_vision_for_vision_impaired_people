# import the necessary packages
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import time
import cv2
import matplotlib.pyplot as plt
import pytesseract
from gtts import gTTS
import text_to_speech
import camera
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
	help="path to input image")
ap.add_argument("-east", "--east", type=str,
	help="path to input EAST text detector")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
	help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=320,
	help="resized image width (should be multiple of 32)")
ap.add_argument("-e", "--height", type=int, default=320,
	help="resized image height (should be multiple of 32)")
args = vars(ap.parse_args())
# calling camera's cam() function that capture image
#camera.cam()
# load the input image and grab the image dimensions
image = cv2.imread(args["image"])
orig = image.copy()
(H, W) = image.shape[:2]

# set the new width and height and then determine the ratio in change
# for both the width and height
(newW, newH) = (args["width"], args["height"])
rW = W / float(newW)
rH = H / float(newH)

# resize the image and grab the new image dimensions
image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]

layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]
# load the pre-trained EAST text detector
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet(args["east"])

# construct a blob from the image and then perform a forward pass of
# the model to obtain the two output layer sets
blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
	(123.68, 116.78, 103.94), swapRB=True, crop=False)
start = time.time()
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)
end = time.time()

# show timing information on text prediction
print("[INFO] text detection took {:.6f} seconds".format(end - start))

# grab the number of rows and columns from the scores volume, then
# initialize our set of bounding box rectangles and corresponding
# confidence scores
(numRows, numCols) = scores.shape[2:4]
rects = []
confidences = []

# loop over the number of rows
for y in range(0, numRows):
	# extract the scores (probabilities), followed by the geometrical
	# data used to derive potential bounding box coordinates that
	# surround text
	scoresData = scores[0, 0, y]
	xData0 = geometry[0, 0, y]
	xData1 = geometry[0, 1, y]
	xData2 = geometry[0, 2, y]
	xData3 = geometry[0, 3, y]
	anglesData = geometry[0, 4, y]

# loop over the number of columns
	for x in range(0, numCols):
		# if our score does not have sufficient probability, ignore it
		if scoresData[x] < args["min_confidence"]:
			continue

		# compute the offset factor as our resulting feature maps will
		# be 4x smaller than the input image
		(offsetX, offsetY) = (x * 4.0, y * 4.0)

		# extract the rotation angle for the prediction and then
		# compute the sin and cosine
		angle = anglesData[x]
		cos = np.cos(angle)
		sin = np.sin(angle)

		# use the geometry volume to derive the width and height of
		# the bounding box
		h = xData0[x] + xData2[x]
		w = xData1[x] + xData3[x]

		# compute both the starting and ending (x, y)-coordinates for
		# the text prediction bounding box
		endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
		endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
		startX = int(endX - w)
		startY = int(endY - h)

		# add the bounding box coordinates and probability score to
		# our respective lists
		rects.append((startX, startY, endX, endY))
		confidences.append(scoresData[x])

# apply non-maxima suppression to suppress weak, overlapping bounding
# boxes
boxes = non_max_suppression(np.array(rects), probs=confidences)
# combine all boxes to create a single bounding box for the entire text
combined_box = np.array(boxes).reshape(-1, 4)
x1 = np.min(combined_box[:, 0])
y1 = np.min(combined_box[:, 1])
x2 = np.max(combined_box[:, 2])
y2 = np.max(combined_box[:, 3])

# scale the bounding box coordinates based on the respective ratios
x1 = int(x1 * rW)
y1 = int(y1 * rH)
x2 = int(x2 * rW)
y2 = int(y2 * rH)

# extract the actual padded ROI
roi = orig[y1:y2, x1:x2]
# 1. Normalization
normalized_image = roi.astype(np.float32) / 255.0
# 2. Image Scaling
target_width = 640  # Desired width
aspect_ratio = target_width / roi.shape[1]
target_height = int(roi.shape[0] * aspect_ratio)
scaled_image = cv2.resize(normalized_image, (target_width, target_height))
# 3. Noise Removal
denoised_image = cv2.GaussianBlur(scaled_image, (5, 5), 0)
# 4. Gray Scale Image
gray_image = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2GRAY)
# Convert to the appropriate data type (CV_8UC1)
gray_image = gray_image.astype(np.uint8)
# threshold the image using Otsu's thresholding method
binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# Convert the binary image to the correct format
binary_image = np.uint8(binary_image)
# Save the binary image as a temporary file
temp_image_path = "temp_binary_image.jpg"
cv2.imwrite(temp_image_path, binary_image)      
# OCR using pytesseract
detected_text = pytesseract.image_to_string(temp_image_path)
os.remove(temp_image_path)
# print the detected text
print("Detected Text:", detected_text)
# Check if the extracted text is not empty
if detected_text:
	# Save the extracted text in a file
	with open("extracted_text.txt", "w") as file:
		file.write(detected_text)
	text_to_speech.speak(detected_text)
# draw the bounding box on the image
cv2.rectangle(orig, (x1, y1), (x2, y2), (0, 255, 0), 2)
# show the output image
image_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
plt.imshow(temp_image_path)
plt.axis("off")
plt.pause(5)
cv2.waitKey(0)

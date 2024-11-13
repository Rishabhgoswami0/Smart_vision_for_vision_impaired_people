# Smart Vision: Assistive Technology for the Visually Impaired

## Table of Contents
1. [Abstract](#abstract)
2. [Introduction](#introduction)
3. [Objectives](#objectives)
4. [System Specifications](#system-specifications)
5. [Working Framework](#working-framework)
6. [Implementation](#implementation)
7. [Methodology](#methodology)
8. [Concept Diagram](#concept-diagram)
9. [Work Flow](#work-flow)
10. [System Testing](#system-testing)
11. [Results](#results)
12. [Features](#features)
13. [Merits and Limitations](#merits-and-limitations)
14. [Future Scope](#future-scope)
15. [Conclusion](#conclusion)

---

### Abstract
Smart Vision is a system designed to empower blind individuals by enabling them to perceive their surroundings through audio feedback. It combines face recognition and text extraction into a single device, allowing users to identify people and read text through a Raspberry Pi-based platform.

---

### Introduction
The Smart Vision system is tailored for blind people to read typed English text and recognize familiar faces. Leveraging the Raspberry Pi, OpenCV, EasyOCR, and text-to-speech technology, the system translates visual information into audio, enhancing interaction and independence for visually impaired individuals.

---

### Objectives
1. Provide a cost-effective and efficient device for reading English text and identifying people.
2. Deliver real-time audio feedback through headphones.
3. Enable hands-free operation via a voice-controlled assistant.

---

### System Specifications

#### Hardware
- **Raspberry Pi 4B** with 8GB RAM
- **Camera** for capturing real-time images
- **Headphones** for audio output
- **Power Bank** for portable use

#### Software
- **Python Libraries**: OpenCV, EasyOCR, SciPy, NumPy, Face Recognition, Pickle
- **Operating System**: Raspberry Pi OS

---

### Working Framework
The Smart Vision system includes two main modules:
1. **Face Recognition**: Detects and identifies faces in real-time and provides audio feedback.
2. **Text Extraction**: Uses OCR to recognize English text in images and converts it to speech.

#### System Block Diagram
To view the system’s block diagram, add an image using the following Markdown syntax:
```markdown
![System Block Diagram](path_to_your_image.png)
```

---

### Implementation

1. **Raspberry Pi Setup**: Install Raspbian OS on an SD card, set up libraries, and configure hardware.
2. **Install Libraries**:
### `install_opencv.sh`

```bash
#!/bin/bash

# Update and upgrade the system
echo "Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y

# Step 1: Install dependencies for OpenCV
echo "Installing build essentials and other dependencies..."
sudo apt-get install -y build-essential cmake unzip pkg-config

echo "Installing image and video libraries..."
sudo apt-get install -y libjpeg-dev libpng-dev libtiff-dev
sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install -y libxvidcore-dev libx264-dev

echo "Installing Python 3 development headers..."
sudo apt-get install -y python3-dev

# Step 2: Download OpenCV and OpenCV_contrib
echo "Downloading OpenCV and OpenCV_contrib..."
cd ~
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.0.0.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.0.0.zip

echo "Unzipping OpenCV files..."
unzip opencv.zip
unzip opencv_contrib.zip

# Step 3: Set up Python virtual environment
echo "Installing pip..."
wget https://bootstrap.pypa.io/get-pip.py
sudo python3 get-pip.py
rm get-pip.py

echo "Installing virtualenv and virtualenvwrapper..."
sudo pip install virtualenv virtualenvwrapper
rm -rf ~/.cache/pip

echo "Configuring virtual environment paths in ~/.profile..."
echo -e "\n## virtualenv and virtualenvwrapper" >> ~/.profile
echo "export WORKON_HOME=$HOME/.virtualenvs" >> ~/.profile
echo "export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3" >> ~/.profile
echo "source /usr/local/bin/virtualenvwrapper.sh" >> ~/.profile
source ~/.profile

echo "Creating a Python 3 virtual environment named 'cv'..."
mkvirtualenv cv -p python3

# Install numpy in virtual environment
echo "Installing numpy in the virtual environment..."
workon cv
pip install numpy

# Step 4: Build OpenCV
echo "Building OpenCV..."
cd ~/opencv-4.0.0/
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D INSTALL_PYTHON_EXAMPLES=ON \
      -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib-4.0.0/modules \
      -D BUILD_EXAMPLES=ON ..

echo "Compiling OpenCV (this may take a while)..."
make -j$(nproc)
sudo make install
sudo ldconfig

# Step 5: Link OpenCV to virtual environment
echo "Linking OpenCV to the virtual environment..."
cd ~/.virtualenvs/cv/lib/python3.*/site-packages/
ln -s /usr/local/python/cv2/python-3.*/cv2.cpython-*-arm-linux-gnueabihf.so cv2.so
cd ~

# Step 6: Install additional Python packages
echo "Installing additional Python packages..."
pip install pickle5 face-recognition scipy numpy imutils opencv-python easyocr SpeechRecognition

echo "OpenCV and required libraries installed successfully!"
```

### How to Run the Script

1. **Save the script** as `install_opencv.sh`.
2. **Make the script executable**:
   ```bash
   chmod +x install_opencv.sh
   ```
3. **Run the script**:
   ```bash
   ./install_opencv.sh
   ```
3. **Voice Assistant Setup**: Configure the text-to-speech engine with Pyttsx3 or Festival TTS.
4. **Running the Application**: Start the application on the Raspberry Pi with connected peripherals.

---

### Methodology

1. **Face Recognition**:
   - **Image Recognition Method**:
It begins by importing necessary libraries such as imutils, face recognition, pickle, cv2 (OpenCV), and custom modules. The code loads the facial encodings from a pre-trained model stored in the encodings. pickle file. The video stream is initialized, and an FPS (frames per second) counter is started. The system listens for user commands through speech-to-text recognition. There is a function that captures images of an unknown person and trains the model. The main loop continuously reads frames from the video stream, resizes them, and detects faces using the face recognition library. It compares facial encodings with known encodings and assigns names to recognized faces. Rectangles are drawn around the faces, and their names are displayed. If a new person is detected, their name is printed and spoken. If an unknown person is detected, a prompt is given to save their images. If the user confirms, the system captures multiple images, saves them in a new folder, and retrains the model. The frames are displayed, and the FPS counter is updated. Once the loop ends, the FPS counter is stopped, and the video stream is released. Finally, the code joins the monitoring thread and displays the elapsed time and approximate FPS. Overall, the code performs face detection, recognition, and user interaction to enhance the facial recognition system's functionality.
   - **Voice Processing Method**:
Speech-to-Text Recognition: The code incorporates a function called   get_user_input() that uses speech-to-text recognition to capture user commands. It listens for speech input and converts it into text. When prompted for a response, the system waits for the user to speak their command. Once the user provides a response, the function returns the recognized text. This enables the user to interact with the system by giving voice commands. For example, the user can respond to prompts such as saving an unknown person's images by saying "yes" or "no".
Text-to-Speech Synthesis: The code utilizes the text-to-speech module to convert text into spoken language. In various instances, the system generates spoken output to provide information or interact with the user. For example, when a new person is identified, their name is printed on the screen and spoken aloud using the text_to_speech.speak() function. Additionally, messages like "An Unknown Person is Detected" and "Do you want to save it, YES or NO" are generated as spoken output to facilitate communication with the user.
2. **Text Extraction**:
   - **Image processing method**:
Necessary packages for image processing, command-line argument parsing, OCR, and camera operations. It initializes an instance of the EasyOCR reader for text recognition and captures an image using the camera. The input image is loaded, resized, and prepared for text detection using the EAST text detector model. Bounding box coordinates and confidence scores are computed based on the model's predictions. Non-maxima suppression is applied to eliminate weak and overlapping bounding boxes, resulting in a set of accurate text regions. The bounding box coordinates are scaled back to the original image size. The region of interest (ROI) containing the text is extracted from the original image based on the scaled bounding box coordinates. Optionally, the ROI can be resized to enhance OCR accuracy. OCR is performed on the resized ROI using EasyOCR, extracting the text from the image. The detected text is stored, and the bounding box is visualized on the original image.
In summary, the code performs image resizing, text detection using an EAST text detector, non-maxima suppression, ROI extraction, OCR using EasyOCR, and visualization of the detected text region. These operations enable the detection and extraction of text from images, facilitating various applications such as text recognition and analysis.
   - **Voice Processing Method**:
The detected text is stored, and the scrible.writing() function is called to write the detected text into a file. This allows the extracted text to be saved and used for further processing. The next is voice processing, It converts the .txt file to an audio output. Here, the text is converted to speech using a speech synthesizer called Festival TTS. The Raspberry Pi has an onboard audio jack, the onboard audio is generated by a PWM output.

---
---

### Concept Diagram 

![BConcept Diagram](https://github.com/Rishabhgoswami0/Smart_vision_for_vision_impaired_people/blob/main/img/cocptflow.png)
### System Testing
1. **Picture Capturing**: Tests camera functionality and quality of image capture.
2. **Face and Text Recognition**: Assesses accuracy of face and text detection.
3. **Text-to-Voice**: Verifies text-to-speech audio clarity and functionality.
### work flow 

![work flow](https://github.com/Rishabhgoswami0/Smart_vision_for_vision_impaired_people/blob/main/img/wrkflow.png)
### Results
The Smart Vision system successfully:
- Recognizes and names familiar faces.
- ![image recognition](https://github.com/Rishabhgoswami0/Smart_vision_for_vision_impaired_people/blob/main/img/imgrecog.png)
- Reads English text and provides audio feedback for visually impaired users.
- ![image recognition](https://github.com/Rishabhgoswami0/Smart_vision_for_vision_impaired_people/blob/main/img/txtrecog.png)
---

### Features
- **Face Recognition**: Enhances social interaction for visually impaired users.
- **Text Reading**: Allows users to read text in real-time.
- **Voice Assistant**: Hands-free control for ease of use.

---

### Merits and Limitations

#### Merits
- **Face Recognition**: Accurate detection and identification.
- **Text Extraction**: Reliable text reading from images.

#### Limitations
- **Face Recognition**: Limited by Raspberry Pi processing power.
- **Text Extraction**: Currently supports only English text.

---

### Future Scope
- **Multilingual Support**: Expand to multiple languages.
- **Improved Design**: Make the device more compact and portable.
- **Additional Features**: Include navigation, object detection, and warning indicators.

---

### Conclusion
Smart Vision empowers visually impaired individuals with an intuitive assistive device, enabling them to recognize faces and read text in real-time. Built on cost-effective hardware, this project emphasizes accessibility and user independence.

---

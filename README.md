# FacePlay--Face-Gender-and-Emotion-detection-using-OpenCV-DLib-and-FisherFace-Recognizer
Goal : Gender and Emotion Detection 

# 1st Step
Face recognition :The act of identifying or verifying a person from a digital image or a video frame. There are many different facial recognition techniques, most of them are based on different nodal points on a human face. 
To build our face recognition system, we will first perform face detection, extract face embeddings from each face, train a face recognition model on the embeddings, and then finally recognize faces in images with OpenCV.

# OPENCV 
OpenCV is an open-source computer vision and machine learning software library created by Intel.
We used it in python to import as cv2

# Haar Cascade Classifier
Haar Cascade is a machine learning object detection algorithm used to identify objects in an image or video based on the concept of features.
Algorithm has three stages – Haar feature Selection, Adaboost training, Cascade of Classifiers.
Haar feature Selection - A Haar Feature considers adjacent rectangular regions at a specific location in a detection window, sums up the pixel intensities in each region and calculates the difference between these sums.
Adaboost Training – selects the best features(out of 160000+) and trains the classifiers that uses them. This algorithm constructs a “strong” classifier as a linear combination of weighted simple “weak” classifiers.
Cascade of Classifiers - Instead of applying all Haar features on a window, the features are grouped into different stages of classifiers and applied one-by-one. If a window fails the first stage, discard it. We don’t consider the remaining features on it. If it passes, apply the second stage of features and continue the process. The window which passes all stages is a face region.

# Dlib Library

Dlib is a modern C++ toolkit containing machine learning algorithms and tools for creating complex software in C++ as well as supports numerous features like Numerical Algorithms, Image processing, threading, GUI’s, data compression and networking.
The facial landmark detector implemented inside dlib produces 68 (x, y)-coordinates that map to specific facial structures. These 68 point mappings were obtained by training a shape predictor on the labeled iBUG 300-W dataset.
We then compute a bounding box over the ROI and visualize the image using different transparent overlays with each facial landmark region highlighted with a different color.

# FisherFace Recognizer
The Eigenface was the first method considered as a successful technique of face recognition. The Eigenface method uses Principal Component Analysis (PCA) to linearly project the image space to a low dimensional feature space.
The Fisherface method is an enhancement of the Eigenface method that it uses Fisher’s Linear Discriminant Analysis (FLDA or LDA) for the dimensionality reduction. 
The LDA maximizes the ratio of between-class scatter to that of within-class scatter, therefore, it works better than PCA for purpose of discrimination. The Fisherface is especially useful when facial images have large variations in illumination and facial expression.

# Timeline of Project work:

# Face and eye detection using Haar classifiers
We will need to load the required XML classifiers .
We will use Haar Cascade Classifiers from OpenCV to identify different regions of the face.
We will take frames from the video input of webcam and convert into grayscale format for further analysis.
Use the function faceCascade.detectMultiScale() to obtain the coordinates enclosing the region of interest (ROI of the face).
Now for each different region of the face identified we will compute a bounding box of varying colors to differentiate among all the parts.
# Face detection and extraction of facial regions using Dlib
Import the necessary packages and construct the argument parser to parse the arguments.
Initialize Dlib's face detector (Histogram Of Object Gradients :HOG-based) and then create the facial landmark predictor.
Load the input image, resize it, and convert it to grayscale and detect the faces in it.
Loop over the face parts individually from the face detections drawing the specific face part.
Extract the ROI of the face region as a separate image and show the particular face part
Visualize all facial landmarks with a transparent overlay.
# FisherFace work
Fisherface recognizer requires every training data to have the same pixel count. This raises a problem because the dataset from KDEF and IMDB does not have uniform size and thus produces error during training. To address this problem, emotion data prep.py and gender data prep.py are created. Both of them use face detection algorithm from face detection.py to detect faces in photos. Then, the picture would be normalized to uniform size (350px x 350px) and saved in grayscale to speed up the training process. 
To train a model for emotion classier, put images of each of the 7 emotions mentioned in data/raw emotion/ . For example, put the images that show happy emotion in data/emotion/happy/ then, run:
	python3 emotion data prep.py
 	python3 train emotion-classifier.py
And similarly we will do it for gender as well.

# Accuracy
Then we tested the model for accuracy against sample images to obtain the result as follows:
Gender : 86.486                  
Emotion :76.468

# Thanks
To my fellow teammates in this project especially deepak bulani and our guide and mentor Mr. Dinesh Sir


import cv2
import numpy as np
import os

if not os.path.exists('output'):
    os.makedirs('output')

#Load images from directory usb_camera_images\2024.7.6.12.30.59
image_array = []

directory = "usb_camera_images\\2024.7.6.13.59.23\\"
#loop through files in directory
for filename in os.listdir(directory):
    #load the file as a cv2 image
    image = cv2.imread(directory + filename)

    #append the image to the image_array
    image_array.append(image)

# Load the pre-trained MobileNet SSD model and the corresponding prototxt file
model_path = 'mobilenet_iter_73000.caffemodel'
prototxt_path = 'deploy.prototxt'

# Read the model and prototxt file from URLs
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Define the class labels the model was trained on
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", 
           "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", 
           "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

for i, image in enumerate(image_array):
    (h, w) = image.shape[:2]

    # Pre-process the image: resize, subtract mean, and scale
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # Loop over the detections
    for j in range(detections.shape[2]):
        confidence = detections[0, 0, j, 2]
        
        # Filter out weak detections by ensuring the confidence is greater than a threshold
        if confidence > 0.2:
            idx = int(detections[0, 0, j, 1])
            box = detections[0, 0, j, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Draw the bounding box around the detected object
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            label = f"{CLASSES[idx]}: {confidence:.2f}"
            cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the output image
    output_image_path = (f'output/image_{i}.jpg')
    cv2.imwrite(output_image_path, image)


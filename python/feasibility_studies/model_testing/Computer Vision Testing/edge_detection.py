import cv2
import numpy as np
import logging
import os
from PIL import Image
import time

#Load images from directory usb_camera_images\2024.7.6.12.30.59
image_array = []

directory = "usb_camera_images/2024.7.6.13.59.23/"
#loop through files in directory
for filename in os.listdir(directory):
    #load the file as a cv2 image
    image = cv2.imread(directory + filename)

    #append the image to the image_array
    image_array.append(image)

#Convert the images to grayscale
gray_image_array = []
for image in image_array:
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image_array.append(gray_image)

#Convert grayscale images to histogram oriented gradients
#Picture resolution is 640x480

img_size = image_array[0].shape[:2]
cell_size = np.array([32, 32])  # h x w in pixels
block_size = np.array([4, 4])  # h x w in cells
win_size = np.array(img_size) // cell_size  # h x w in cells

n_bins = 9

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

people_images = []

for image in image_array:
    # Detect people in the image
    locations, confidence = hog.detectMultiScale(image)
    
    # Draw rectangles around the detected people
    for (x, y, w, h) in locations:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 5)
    
    #Append image to people_images array if people are detected
    # if len(locations) > 0:
    people_images.append(image)
 
# Display the image with detected people
counter = 0
time_of_last_frame = time.time()
while True:
    if time.time() - time_of_last_frame > 1/5:
        cv2.imshow("People Detection", people_images[counter])
        counter += 1
        counter = counter % len(people_images)
        time_of_last_frame = time.time()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()








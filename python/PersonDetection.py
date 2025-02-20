import torch
from torchvision import models
import cv2
# import numpy as np

model_confidence_threshold = 0.8

# Load the model
model = models.detection.fasterrcnn_resnet50_fpn(weights=models.detection.faster_rcnn.FasterRCNN_ResNet50_FPN_Weights.DEFAULT) #type: ignore
model.eval()

#Function to tell whether a person is in an image or not. Expects a numpy input/opencv image **in RGB format I think**
def detect_person(image):
    # Load and preprocess image using OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0 #(H,C,W) to (C,H,W) to tensor and normalize to [0,1]

    # Perform detection
    with torch.no_grad():
        predictions = model(image)
    
    # SSD MobileNet returns predictions where first element contains boxes, labels, and scores
    # COCO dataset label 1 corresponds to 'person'
    labels = predictions[0]['labels']
    scores = predictions[0]['scores']
    
    # Check if any detection is a person (label 1) with confidence > 0.5
    for label, score in zip(labels, scores):
        if label == 1 and score > model_confidence_threshold:
            return True
    
    return False

if __name__ == "__main__":
    import cv2
    # Loop through loading images in the test_set folder and checking if a person is in them
    for i in range(2, 42):
        image = cv2.imread(f"python/test_set/capture_{i}.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #Need to comment out the one in the function if running this in main
        print(f"capture_{i}.jpg: {detect_person(image)}")
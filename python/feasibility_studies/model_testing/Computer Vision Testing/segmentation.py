import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, sam_model_registry
from pycocotools import mask as mask_utils

image = cv2.imread("usb_camera_images/2024.7.6.12.30.59/image_0.jpg")

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
sam = sam_model_registry["vit_h"](checkpoint=r"C:\Users\user\Documents\CV Models\sam_vit_h_4b8939.pth")
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(image)
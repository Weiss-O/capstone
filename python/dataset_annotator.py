# This script will help me take a directory full of images and use it
# to create an annotated test set that the model testing script can
# understand.

import cv2
import os
import yaml
import argparse

# Global variables
annotations = []
drawing = False
start_x, start_y = -1, -1
current_image = None
current_annotations = []
image_list = []
image_index = 0
baseline_image = None

def draw_rectangle(event, x, y, flags, param):
    global start_x, start_y, drawing, current_image, current_annotations

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp_img = current_image.copy()
            cv2.rectangle(temp_img, (start_x, start_y), (x, y), (0, 255, 0), 2)
            cv2.imshow("Annotation", temp_img)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_x, end_y = x, y
        bbox = (min(start_x, end_x), min(start_y, end_y), abs(end_x - start_x), abs(end_y - start_y))
        current_annotations.append(bbox)
        cv2.rectangle(current_image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
        cv2.imshow("Annotation", current_image)

def annotate_images(directory, baseline_photo=None):
    global image_list, image_index, current_image, current_annotations, baseline_image
    
    image_list = sorted([f for f in os.listdir(directory) if f.lower().endswith(('png', 'jpg', 'jpeg'))])
    if not image_list:
        print("No images found in directory.")
        return
    
    if baseline_photo and baseline_photo in image_list:
        image_list.remove(baseline_photo)
        image_list.insert(0, baseline_photo)
    
    cv2.namedWindow("Annotation")
    cv2.setMouseCallback("Annotation", draw_rectangle)
    
    baseline_path = os.path.join(directory, image_list[0])
    baseline_image = cv2.imread(baseline_path)
    if baseline_image is not None:
        baseline_image = resize_image(baseline_image)
        cv2.namedWindow("Baseline")
        cv2.imshow("Baseline", baseline_image)
    
    for image_name in image_list:
        image_path = os.path.join(directory, image_name)
        current_image = cv2.imread(image_path)
        current_annotations = []
        
        if current_image is None:
            print(f"Error loading {image_path}")
            continue
        
        current_image = resize_image(current_image)
        cv2.imshow("Annotation", current_image)
        key = cv2.waitKey(0)  # Wait for user input
        
        if key == ord('n'):
            scaled_annotations = scale_annotations_back(current_annotations, current_image.shape[1], image_path)
            annotations.append({"photo_path": image_path, "annotations": sorted(scaled_annotations)})

    cv2.destroyAllWindows()
    save_annotations(directory)

def resize_image(image, width=900):
    height = int(image.shape[0] * (width / image.shape[1]))
    return cv2.resize(image, (width, height))

def scale_annotations_back(annotations, current_width, image_path):
    original_image = cv2.imread(image_path)
    scale_factor = original_image.shape[1] / current_width
    return [(int(x * scale_factor), int(y * scale_factor), int(w * scale_factor), int(h * scale_factor)) for (x, y, w, h) in annotations]


def save_annotations(directory):
    dataset = {"images": annotations}
    output_file = os.path.join(directory, "dataset.yaml")
    with open(output_file, 'w') as f:
        yaml.dump(dataset, f, default_flow_style=False)
    print(f"Annotations saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Photo Annotation Script")
    parser.add_argument("-directory_path", required=True, help="Path to the directory containing images")
    parser.add_argument("-baseline_photo_name", required=False, help="Optional baseline photo filename")
    args = parser.parse_args()
    
    annotate_images(args.directory_path, args.baseline_photo_name)

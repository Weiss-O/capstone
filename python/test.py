import numpy as np
import matplotlib.pyplot as plt
import cv2
import random

# Load the image
image_path = "python/test_images/fr_test.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for correct color display in Matplotlib

# Define points
points = [(323, 900), (32, 587), [149, 572], (1770, 545), (1694, 474), (1126, 395), (528, 361), (1038, 360)]

# Plot the image
plt.figure(figsize=(10, 6))
plt.imshow(image)
# Plot points
for point in points:
    x, y = point if isinstance(point, tuple) else (point[0], point[1])  # Handle numpy arrays
    random_color = (random.random(), random.random(), random.random())
    plt.scatter(x, y, color=random_color, s=50, label=f'({x}, {y})' if 'label' not in locals() else "")

plt.legend()
plt.axis("off")
plt.show()

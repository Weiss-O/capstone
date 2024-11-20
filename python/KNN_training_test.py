import cv2 as cv
import os
from matplotlib.widgets import Slider

# Folder paths
training_folder = r'KNN_training'
testing_folder = r'KNN_testing'

# Initialize KNN background subtractor
backSub = cv.createBackgroundSubtractorMOG2()

# Train the background subtractor using all images in the training folder
for filename in sorted(os.listdir(training_folder)):
    file_path = os.path.join(training_folder, filename)
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        frame = cv.imread(file_path)
        if frame is None:
            print(f"Warning: Unable to load {filename} in training folder.")
            continue
        # Apply the background subtractor to update the background model
        _ = backSub.apply(frame)

# Disable background updates to freeze the model
backSub.setDetectShadows(False)  # Optional: disable shadow detection if shadows aren't relevant
backSub.setHistory(0)  # Effectively stops the model from updating
import matplotlib.pyplot as plt

# List of testing images
testing_images = [os.path.join(testing_folder, f) for f in sorted(os.listdir(testing_folder)) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Function to update the displayed image
def update(val):
    idx = int(slider.val)
    file_path = testing_images[idx]
    frame = cv.imread(file_path)
    if frame is None:
        print(f"Warning: Unable to load {file_path} in testing folder.")
        return

    # Apply the background subtractor to detect changes without updating the model
    fgMask = backSub.apply(frame, learningRate=0)  # Set learningRate=0 to prevent updates

    # Update the plot with the new image and mask
    ax1.imshow(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    ax2.imshow(fgMask, cmap='gray')
    fig.canvas.draw_idle()

# Create the plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
plt.subplots_adjust(bottom=0.25)

# Initial display
initial_frame = cv.imread(testing_images[0])
initial_fgMask = backSub.apply(initial_frame, learningRate=0)
ax1.imshow(cv.cvtColor(initial_frame, cv.COLOR_BGR2RGB))
ax1.set_title('Testing Image')
ax2.imshow(initial_fgMask, cmap='gray')
ax2.set_title('Foreground Mask')

# Slider for scrubbing through images
ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider = Slider(ax_slider, 'Image', 0, len(testing_images) - 1, valinit=0, valstep=1)
slider.on_changed(update)

plt.show()

# Cleanup
cv.destroyAllWindows()

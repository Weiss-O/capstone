"""
Script for visualizing the detection results and the camera pointing math.
"""

import Camera #Might create a cirular import, not sure
import numpy as np

##VISUALIZATION CODE##
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import datetime
import yaml

with open('config.yaml') as file:
    config = yaml.safe_load(file)

def plot_detections(detections, baseline_image, test_image):
    """
    Function to generate plot visualizing detections

    Args:
        detections: List of raw detections from the detector
        baseline_image: cv2 image object
        test_image: cv2 image object

    Returns:
        None
    """
    #Create matplotlib figure
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    #Plot the baseline image with the baseline masks
    ax1.imshow(baseline_image)
    baselineMasks = [detection.baselineMask for detection in detections]

    show_masks(ax1, baselineMasks)

    #Plot the test image with the test masks
    ax2.imshow(test_image)
    testMasks = [detection.testMask for detection in detections]

    show_masks(ax2, testMasks)

    #Save the plot as an image
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f"output/detections_{date}.png")

def show_mask(mask, ax, random_color=True, borders = False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_masks(ax, masks, borders=False):
    if masks:  # Check if masks list is not empty
        for i, mask in enumerate(masks[:3]):
            if mask is not None:  # Check if individual mask is not None
                show_mask(mask, ax, borders=borders)

        # if len(scores) > 1:
        #     ax.set_title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    # ax.axis('off')

def ft_to_mm(ft):
    return ft*304.8

room_width = ft_to_mm(15)
room_height = ft_to_mm(10)

def plot_camera(detections, camera_pos, baseline, test):
    """
    Function to visualize in 3D the camera FOV and the calculated projection rays.

    Args:
        detections: List of Detection objects

    Returns:
        None
    """
    camera = Camera.CameraStandIn(config["camera_settings"])
    camera.update_ReferenceFrame(camera_pos[0], camera_pos[1])

    #Create matplotlib figure
    fig = plt.figure(figsize=(20, 10))
    ax3d = fig.add_subplot(221, projection='3d')
    ax2d = fig.add_subplot(222)
    axBaseline = fig.add_subplot(223)
    axTest = fig.add_subplot(224)

    # Add lines to pass through the origin for each axis
    ax3d.plot([-1000, 1000], [0, 0], [0, 0], 'k--', alpha=0.5)  # X-axis
    ax3d.plot([0, 0], [-1000, 1000], [0, 0], 'k--', alpha=0.5)  # Y-axis
    ax3d.plot([0, 0], [0, 0], [-1000, 1000], 'k--', alpha=0.5)  # Z-axis

    # Setting axis limits
    ax3d.set_xlim([0, room_width])
    ax3d.set_ylim([0, room_width])
    ax3d.set_zlim([-room_height, 0])
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')

    ax2d.set_xlim([0, camera.reference_frame.resX])
    ax2d.set_ylim([-camera.reference_frame.resY, 0])
    ax2d.set_xlabel('X')
    ax2d.set_ylabel('Y')
    
    #Set titles
    ax3d.set_title('3D Representation of Camera FOV')
    ax2d.set_title('2D Representation of Camera FOV')

    # Define the four corners of the image frame in the camera image frame
    image_corners = np.array([
        [0, 0, 1],  # Bottom-left
        [camera.reference_frame.resX, 0, 1],   # Bottom-right
        [camera.reference_frame.resX, camera.reference_frame.resY, 1],    # Top-right
        [0, camera.reference_frame.resY, 1]    # Top-left
    ])

    #project out at a distance of 100mm from the camera in the camera z direction
    cone_depth = 500
    cam_corners = [np.append(camera.reference_frame.image_to_cam(corner, cone_depth), 1) for corner in image_corners]
    
    #Transform the corners to the world frame
    transformed_corners = [np.dot(camera.reference_frame.T_cam_world, corner) for corner in cam_corners]
    transformed_camera_origin = np.dot(camera.reference_frame.T_cam_world, np.array([0, 0, 0, 1]))
    
    # Plot the vector from the origin to the world point
    ax3d.plot([0, transformed_camera_origin[0]], [0, transformed_camera_origin[1]], [0, transformed_camera_origin[2]], 'g-')

    #Plot the camera FOV
    ax3d.plot(
        np.array([[transformed_camera_origin[0], corner[0]] for corner in transformed_corners]).flatten(),
        np.array([[transformed_camera_origin[1], corner[1]] for corner in transformed_corners]).flatten(),
        np.array([[transformed_camera_origin[2], corner[2]] for corner in transformed_corners]).flatten(),
        'b-'
    )

    detection_boxes = [detection.get_as_array() for detection in detections]
    #Center of detected object in camera image
    camera_objs = [[box[0] + box[2]/2, box[1] + box[3]/2, 1] for box in detection_boxes]

    ax2d.scatter([cam_obj[0] for cam_obj in camera_objs],
                             [-cam_obj[1] for cam_obj in camera_objs],
                             c='r',
                             s=100)
    
    #Back Calculate estimated theta and phi
    spherical_coords = np.array([camera.calculate_pointing_ray(obj, degrees=False) for obj in camera_objs])

    rays = np.array([[
        np.cos(c[0])*np.cos(c[1]),
        np.sin(c[0])*np.cos(c[1]),
        -np.sin(c[1])] for c in spherical_coords])*ft_to_mm(15)
    
    ax3d.plot(
        np.array([[0, ray[0]] for ray in rays]).flatten(),
        np.array([[0, ray[1]] for ray in rays]).flatten(),
        np.array([[0, ray[2]] for ray in rays]).flatten(),
        'black'
    )

    #Plot the baseline image in the bottom left
    axBaseline.imshow(baseline)

    axBaseline.set_title('Baseline Image')
    axBaseline.axis('off')

    #Plot the test image in the bottom right
    axTest.imshow(test)

    #Plot the camera_objs points on the test image
    axTest.scatter([cam_obj[0] for cam_obj in camera_objs],
                   [cam_obj[1] for cam_obj in camera_objs],
                   c='r',
                   s=100)


    axTest.set_title('Test Image w/ Detections')
    axTest.axis('off')

    #Set the overall plot title
    plt.suptitle(f'Camera Position (Theta, Phi): ({camera_pos[0]}, {camera_pos[1]})')
    #save the plot as an image
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    plt.savefig(f"output/camera_{date}_{camera_pos[0]}_{camera_pos[1]}.png")
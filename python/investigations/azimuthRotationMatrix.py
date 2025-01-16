import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider

# Create a point in the camera coordinate frame
camera_point = np.array([0, 0, 1, 1])

# Image frame dimensions
width = 0.5  # Example width in arbitrary units
height = 0.3  # Example height in arbitrary units

# Define the four corners of the image frame in the camera coordinate frame
image_corners = np.array([
    [-width / 2, -height / 2, 1, 1],  # Bottom-left
    [width / 2, -height / 2, 1, 1],   # Bottom-right
    [width / 2, height / 2, 1, 1],    # Top-right
    [-width / 2, height / 2, 1, 1]    # Top-left
])

# Function to compute the rotation matrix
def compute_rotation_matrix(theta, phi):
    theta_rad = np.radians(theta)
    phi_rad = np.radians(phi)
    R = np.array([
        [-np.sin(theta_rad), np.cos(theta_rad), 0, 0],
        [np.sin(phi_rad) * np.cos(theta_rad), np.sin(phi_rad) * np.sin(theta_rad), np.cos(phi_rad), 0],
        [np.cos(phi_rad) * np.cos(theta_rad), np.cos(phi_rad) * np.sin(theta_rad), -np.sin(phi_rad), 0],
        [0, 0, 0, 1]
    ])

    #Return inverse
    return np.linalg.inv(R)

# Initialize the rotation parameters
theta = 0
phi = 0

# Initial rotation matrix
R = compute_rotation_matrix(theta, phi)
R2 = compute_rotation_matrix(2*theta, phi)

# Transform the point to the world frame
def transform_point(point, rotation_matrix):
    return np.dot(rotation_matrix, point)

# Initialize the transformed point
world_point = transform_point(camera_point, R)

# Transform the corners to the world frame
transformed_corners = [transform_point(corner, R) for corner in image_corners]

# Create the figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Set up the plot limits and labels
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Add lines to pass through the origin for each axis
ax.plot([-1, 1], [0, 0], [0, 0], 'k--', alpha=0.5)  # X-axis
ax.plot([0, 0], [-1, 1], [0, 0], 'k--', alpha=0.5)  # Y-axis
ax.plot([0, 0], [0, 0], [-1, 1], 'k--', alpha=0.5)  # Z-axis

# Plot the vector from the origin to the world point
vector, = ax.plot([0, world_point[0]], [0, world_point[1]], [0, world_point[2]], 'r-')

#Plot the camera frame
image_frame, = ax.plot(
    [corner[0] for corner in transformed_corners] + [transformed_corners[0][0]], 
    [corner[1] for corner in transformed_corners] + [transformed_corners[0][1]],
    [corner[2] for corner in transformed_corners] + [transformed_corners[0][2]],
    'b-'
)

# Create sliders for theta and phi
ax_theta = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_phi = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')

slider_theta = Slider(ax_theta, 'Theta', 0, 360, valinit=theta)
slider_phi = Slider(ax_phi, 'Phi', 0, 360, valinit=phi)

# Update function for the sliders
def update(val):
    global R
    theta = slider_theta.val
    phi = slider_phi.val
    R = compute_rotation_matrix(theta, phi)
    world_point = transform_point(camera_point, R)
    
    # Update the vector
    vector.set_data([0, world_point[0]], [0, world_point[1]])
    vector.set_3d_properties([0, world_point[2]])
    
    # Update the image frame corners
    transformed_corners = [transform_point(corner, R) for corner in image_corners]
    image_frame.set_data(
        [corner[0] for corner in transformed_corners] + [transformed_corners[0][0]],  # Close the loop
        [corner[1] for corner in transformed_corners] + [transformed_corners[0][1]]
    )
    image_frame.set_3d_properties(
        [corner[2] for corner in transformed_corners] + [transformed_corners[0][2]]
    )
    fig.canvas.draw_idle()

# Connect sliders to update function
slider_theta.on_changed(update)
slider_phi.on_changed(update)

plt.show()

import numpy as np
from mpl_toolkits.mplot3d import Axes3D

#Variables for required Camera Positions for 

camera_fov_width = 60 #horizontal FOV of camera in degrees
camera_fov_height = 47 #vertical FOV of camera in degrees

wall_distance = 0.2 #distance of wall to positioning system center of rotation in meters
r_camera = 0.1 #distance of camera to positioning system center of rotation in meters

theta_offset_cam_pos = 0 #offset between 0 of positioning system and camera center in degrees
phi_offset_cam_pos = 0 #offset between 0 of positioning system and camera center in degrees

theta_offset_wall_pos = 0 #offset between 0 of positioning system and right wall in degrees
phi_offset_ceiling_pos = 0 #offset between 0 of positioning system and ceiling in degrees

phi0 = phi_offset_ceiling_pos - phi_offset_cam_pos - 90 + camera_fov_height/2 #Angle to position camera with bottom of FOV at corner of room
theta0 = theta_offset_wall_pos - theta_offset_cam_pos + 45 #Angle to center camera wrt two adjacent walls

theta1 = theta_offset_wall_pos - theta_offset_cam_pos + 90 - camera_fov_width/2
phi1 = phi0 + camera_fov_height

theta2 = theta_offset_wall_pos - theta_offset_cam_pos + camera_fov_width/2
phi2 = phi0 + camera_fov_height

w_room = 4.572 #width of room in meters
h_room = 3.048 #height of room in meters

camera_resolution = [1920, 1080] #resolution of camera in pixels

#Array to store theta_phi positions of camera
camera_positions = [[theta0, phi0], [theta1, phi1], [theta2, phi2]]


import matplotlib.pyplot as plt

# Function to convert spherical coordinates to cartesian coordinates
def spherical_to_cartesian(r, theta, phi):
    theta_rad = np.deg2rad(theta)
    phi_rad = np.deg2rad(phi)
    x = r * np.cos(phi_rad) * np.cos(theta_rad)
    y = r * np.cos(phi_rad) * np.sin(theta_rad)
    z = r * np.sin(phi_rad)
    return x, y, z

#Function to create a rectangle representing the projection of the camera FOV
def create_camera_fov(theta, phi, r, fov_width, fov_height):
    x, y, z = spherical_to_cartesian(r, theta, phi)
    x1, y1, z1 = spherical_to_cartesian(r, theta + fov_width, phi)
    x2, y2, z2 = spherical_to_cartesian(r, theta + fov_width, phi + fov_height)
    x3, y3, z3 = spherical_to_cartesian(r, theta, phi + fov_height)
    return [[x, y, z], [x1, y1, z1], [x2, y2, z2], [x3, y3, z3], [x, y, z]]

#Function to generate projection matrix for camera given its position and orientation
def generate_projection_matrix(theta, phi, x0, y0, z0, camera_fov, camera_resolution):
    #Define camera position and orientation
    camera_position = np.array([x0, y0, z0])
    camera_orientation = np.array([np.cos(np.deg2rad(phi)) * np.cos(np.deg2rad(theta)), np.cos(np.deg2rad(phi)) * np.sin(np.deg2rad(theta)), np.sin(np.deg2rad(phi))])
    #Define camera FOV
    camera_fov = np.array([[np.cos(np.deg2rad(theta)), -np.sin(np.deg2rad(theta)), 0, 0], [np.sin(np.deg2rad(theta)), np.cos(np.deg2rad(theta)), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    #Transformation from homogenoug pixel coordinates to camera coordinates
    pixel_to_camera = np.array([[1, 0, -camera_resolution[0]/2, 0], [0, 1, -camera_resolution[1]/2, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    #Transformation from camera coordinates to world coordinates
    camera_to_world = np.array([[camera_orientation[0], camera_orientation[1], camera_orientation[2], camera_position[0]], [0, 0, 0, camera_position[1]], [0, 0, 0, camera_position[2]], [0, 0, 0, 1]])
    #Projection matrix
    projection_matrix = camera_to_world @ camera_fov @ pixel_to_camera

    return projection_matrix

#Function to take pixel value and convert to world coordinates
def pixel_to_world(pixel_value, projection_matrix):
    pixel_value = np.array([pixel_value[0], pixel_value[1], 1])
    world_coordinates = np.linalg.inv(projection_matrix) @ pixel_value
    return world_coordinates

#Plot projection of 0, 0 pixel for each camera position
for theta, phi in camera_positions:
    x, y, z = spherical_to_cartesian(r_camera, theta, phi)
    projection_matrix = generate_projection_matrix(theta, phi, x, y, z, camera_fov_width, camera_resolution)
    world_coordinates = pixel_to_world([0, 0], projection_matrix)
    plt.plot(world_coordinates[0], world_coordinates[1], marker = 'x')


# Origin point
origin = np.array([w_room - wall_distance, w_room - wall_distance, h_room])

# Create a new figure for 3D plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot rays for each camera position
for theta, phi in camera_positions:
    x0, y0, z0 = spherical_to_cartesian(r_camera, theta, phi)
    x1, y1, z1 = spherical_to_cartesian(10, theta, phi)
    ax.plot([origin[0] + x0, origin[0] + x1], [origin[1] + y0, origin[1] + y1], [origin[2] + z0, origin[2] + z1], marker = 'o')

# Plot camera FOV for each camera position
for theta, phi in camera_positions:
    fov = create_camera_fov(theta, phi, r_camera, camera_fov_width, camera_fov_height)
    ax.plot([p[0] for p in fov], [p[1] for p in fov], [p[2] for p in fov])

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

#set axis limits to 15 ft x 15 ft x 10 ft
ax.set_xlim([0, w_room])
ax.set_ylim([0, w_room])
ax.set_zlim([0, h_room])

# Show plot
plt.show()

"""
class Camera_Position:
    def __init__(self, theta: float, phi: float):
        self.theta = theta
        self.phi = phi
"""
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

f = 4.74e-3 #focal length of camera in meters
w_sensor = 6.45e-3 #width of camera sensor in meters
h_sensor = 3.6e-3 #height of camera sensor in meters

#Projection matrix from camera cartesian frame to camera image frame
T_cam_sensor = np.array([[f, 0, 0, 0], [0, f, 0, 0], [0, 0, 1, 0]]) #x', y', z' = T_cam_sensor * x, y, z, 1

#Array of 3D points to test projection
points = np.array([[0, 1, 1], [1, 0, 1], [0.1, 0.1, 1], [-0.1, -0.1, 1]])

#Project points to camera image frame
points_hom = np.hstack((points, np.ones((points.shape[0], 1))))
points_image = T_cam_sensor @ points_hom.T

#Recover image coordinates:
points_image = points_image[:2, :] / points_image[2, :] #Normalize by z'


import matplotlib.pyplot as plt

#Plot 3D points and their projections
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


#Draw a ray from origin to each point in world coordinates
for point in points:
    ax.plot([0, point[0]], [0, point[1]], [0, point[2]], color='r', marker='o')

#ax.scatter(points[:,0], points[:,1], points[:,2], color='r')
ax.scatter(points_image[0,:], points_image[1,:], f, color='b')

#Label axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()

#function to project point in image frame onto sphere of radius r in cartesian
def project_point(point, r):
    x_p = point[0]
    y_p = point[1]
    point = np.array([x_p, y_p, f])
    point = point / np.linalg.norm(point)
    point = point * r
    return point

#Generate a grid of xy points spaced about origin in image frame
x = np.linspace(-w_sensor/2, w_sensor/2, 20)
y = np.linspace(-h_sensor/2, h_sensor/2, 20)
X, Y = np.meshgrid(x, y)
points = np.vstack((X.ravel(), Y.ravel())).T

#Project points to sphere
points_sphere = np.array([project_point(point, 1) for point in points])

ax.scatter(points_sphere[:,0], points_sphere[:,1], points_sphere[:,2], color='r')

#Plot points on sphere
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#Plot rays from origin to points on sphere
for point in points_sphere:
    ax.plot([0, point[0]], [0, point[1]], [0, point[2]], color='r', marker='o')

ax.scatter(points[:,0], points[:,1], f, color='b')

#Label axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

#Find required range in each axis
max_range = np.max(np.abs(points_sphere))
ax.set_xlim([-max_range/2, max_range/2])
ax.set_ylim([-max_range/2, max_range/2])
ax.set_zlim([0, max_range])

plt.show()

#Define camera position and orientation
camera_position = np.array([1, 1, 1])
camera_orientation = np.array([1, 1, -1]) / np.linalg.norm(np.array([1, 1, -1]))
camera_orientation = camera_orientation.flatten()

#Matrix translation from world to camera coordinates
T_world_camera = np.array([
    [camera_orientation[0], -camera_orientation[1], -camera_orientation[2], -camera_position[0]], 
    [camera_orientation[1], camera_orientation[0], 0, -camera_position[1]], 
    [camera_orientation[2], 0, camera_orientation[0], -camera_position[2]], 
    [0, 0, 0, 1]
])

#Matrix translation from camera to world coordinates
T_camera_world = np.linalg.inv(T_world_camera)

#Convert sphere points to world coordinates
points_world = np.array([T_camera_world @ np.hstack((point, 1)) for point in points_sphere])
camera_origin = T_camera_world @ np.array([0, 0, 0, 1])

#Plot points in world coordinates
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#Plot rays from camera origin to points in world coordinates
for point in points_world:
    ax.plot([camera_origin[0], point[0]], [camera_origin[1], point[1]], [camera_origin[2], point[2]], color='r', marker='o')

#Label axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

#Find required range in each axis
max_range = np.max(np.abs(points_world))
ax.set_xlim([-max_range/2, max_range/2])
ax.set_ylim([-max_range/2, max_range/2])
ax.set_zlim([0.5, 3.5])

plt.show()
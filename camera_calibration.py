
# coding: utf-8

# # Camera calibration: adjusting camera lense distortion
# by Uki D. Lucas

# ## Motivation
# 
# The raw camera data usually has a certain level of distortion caused by lense shape, this is especially pronounced on the edges of the image. The correction is essential in applications like image recognition used in autonomous vehicles, robotics and even in 3D printing.

# ## Solution approach
# 
# The common solution is to compare a **known shape object e.g. a chessboard** with the image taken, then calculate this **specific camera's** adjustment parameters that then can be applied to every frame taken by the camera. If the camera changes, the parameters have to be recalibrated.

# In[1]:

#!/usr/bin/python3
import numpy as np
import cv2
import glob
import json
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# ## Useful helper method

# In[2]:

def plot_images(left_image, right_image):
    plt.figure(figsize=(20,10))
    plot_image = np.concatenate((left_image, right_image), axis=1)
    plt.imshow(plot_image)
    plt.show() 


# ## The pattern will look for "inner corners" e.g. black touching the black square

# In[3]:

nx = 9 # horizontal
ny = 6 # vertical


# In[4]:

# read a list of files using a parern
images = glob.glob("camera_cal/calibration*.jpg") # e.g. calibration19.jpg
print("found", len(images), "images" )


# In[5]:

# Initialise arrays

# Object Points: 3d point in real world space
object_point_list = []

#Image Points: 2d points in image plane.
image_points_list = []


# In[6]:

# Generate 3D object points
object_points = np.zeros((nx*ny, 3), np.float32)
object_points[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

print("first 5 elements:\n", object_points[0:5])


# In[7]:

# see: http://docs.opencv.org/trunk/dc/dbb/tutorial_py_calibration.html

termination_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
chessboard_dimentions = (nx, ny)
for file_name in images:
    image_original = cv2.imread(file_name)
    
    # convert BGR image to gray-scale
    image_gray = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    # Paramters:
    # - image_gray
    # - the chessboard to be used is 9x6
    # - flags = None
    has_found, corners = cv2.findChessboardCorners(image_gray, chessboard_dimentions, None)
    print("processing:", file_name)
    if has_found == True:
        # fill in ObjectPoints
        object_point_list.append(object_points)

        corners2 = cv2.cornerSubPix(image_gray, corners, (11,11), (-1,-1), termination_criteria)
        # fill in ImagePoints
        image_points_list.append(corners2)

        # Draw and display the corners
        # I have to clone/copy the image because cv2.drawChessboardCorners changes the content
        image_corners = cv2.drawChessboardCorners(image_original.copy(), chessboard_dimentions, corners2, has_found)
          
        plot_images(image_original, image_corners)
    else:
        print("The", chessboard_dimentions, "chessboard pattern was not found in file", file_name)
        plt.figure()
        plt.imshow(image_original)
        plt.show()
        


# # Calibrate using points

# In[8]:

# Returns:
# - camera matrix
# - distortion coefficients
# - rotation vectors
# - translation vectors
ret, matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
    object_point_list, 
    image_points_list, 
    image_gray.shape[::-1], 
    None, 
    None)


# ## This instead of concatinating corners, STRETCH them

# In[9]:

image_dimentions = image_gray.shape[:2] # height, width
new_matrix, roi = cv2.getOptimalNewCameraMatrix(matrix, distortion, image_dimentions, 1, image_dimentions)


# # Remove the distortion form the images

# In[10]:

for file_name in images:
    print("processing:", file_name)
    image_original = cv2.imread(file_name)

    image = cv2.undistort(image_original, matrix, distortion, None, None) #new_matrix 
    plot_images(image_original, image)
    
    image = cv2.undistort(image_original, matrix, distortion, None, new_matrix ) 
    plot_images(image_original, image)


# In[12]:

# save to disk
if "calibration2.jpg" in file_name:
    cv2.imwrite('image_undistorted_2.png', image)


# In[ ]:




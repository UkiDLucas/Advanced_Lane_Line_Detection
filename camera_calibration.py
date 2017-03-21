
# coding: utf-8

# # Camera calibration: adjusting camera lense edge distortion
# by Uki D. Lucas

# ## Motivation
# 
# The raw camera data usually has a certain level of distortion caused by lense shape, this is especially pronounced on the edges of the image. The correction is essential in applications like image recognition where mainitaining the shape is essential, especially in autonomous vehicles, robotics and even in 3D printing.

# ## Solution approach
# 
# The common solution is to compare a **known shape object e.g. a chessboard** with the image taken, then calculate this **specific camera's** adjustment parameters that then can be applied to every frame taken by the camera. If the camera changes (camera lense type, resolution, image size) in the set up, the new chessboard images have to be taken and calibration process repeated.
# 
# The angle of chessboard in relation to the camera does not matter, in fact the sample pictures should be taken from various angles. Only the pictures with **whole chessboard showing** will work.
# 
# You should take **many, at least 20 sample images**, any less and the learning process renders images that are not a well corrected.

# # Desired result (spoiler alert!)
# 
# For people who have no patience to read the whole paper I am including the final result:
# 
# 
# <img src="example_calibration.png" />

# In[1]:

#!/usr/bin/python3
import numpy as np # used for lists, matrixes, etc.
import cv2 # we will use OpenCV library
get_ipython().magic('matplotlib inline')


# ## Get list of sample chessboard images by this camera

# In[2]:

# read a list of files using a parern
import glob
image_file_names = glob.glob("camera_cal/calibration*.jpg") # e.g. calibration19.jpg
print("found", len(image_file_names), "calibration image samples" )
print("example", image_file_names[0])


# ## Useful helper method

# In[3]:

def plot_images(left_image, right_image):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(20,10))
    plot_image = np.concatenate((left_image, right_image), axis=1)
    plt.imshow(plot_image)
    plt.show() 


# In[4]:

def get_sample_gray(image_file_name: str):
    import cv2
    image_original = cv2.imread(image_file_name)
    # convert BGR image to gray-scale
    image_gray = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
    return image_original, image_gray
    


# ## The pattern will look for "inner corners" e.g. black touching the black square

# In[5]:

# Chessboard dimentsions
# nx = 9 # horizontal
# ny = 6 # vertical

def find_inside_corners(image_file_names: list, nx: int=9, ny: int=6):
    # Initialise arrays

    # Object Points: 3d point in real world space
    object_point_list = []

    #Image Points: 2d points in image plane.
    image_points_list = []

    # Generate 3D object points
    object_points = np.zeros((nx*ny, 3), np.float32)
    object_points[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    #print("first 5 elements:\n", object_points[0:5])
    # see: http://docs.opencv.org/trunk/dc/dbb/tutorial_py_calibration.html

    termination_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    chessboard_dimentions = (nx, ny)
    import matplotlib.pyplot as plt
    for image_file_name in image_file_names:
        
        print("processing image:", image_file_name)
        image_original, image_gray = get_sample_gray(image_file_name)

        # Find the chess board corners
        # Paramters:
        # - image_gray
        # - the chessboard to be used is 9x6
        # - flags = None
        has_found, corners = cv2.findChessboardCorners(image_gray, chessboard_dimentions, None)
        
        if has_found == True:
            # fill in ObjectPoints
            object_point_list.append(object_points)

            corners2 = cv2.cornerSubPix(image_gray, corners, (11,11), (-1,-1), termination_criteria)
            # fill in ImagePoints
            image_points_list.append(corners2)

            # Draw and display the corners
            # I have to clone/copy the image because cv2.drawChessboardCorners changes the content
            image_corners = cv2.drawChessboardCorners(
                image_original.copy(), 
                chessboard_dimentions, 
                corners2, 
                has_found)

            plot_images(image_original, image_corners)
        else: # not has_found
            print("The", chessboard_dimentions, 
                  "chessboard pattern was not found, most likely partial chessboard showing")
            plt.figure()
            plt.imshow(image_original)
            plt.show()
        # end if
    # end for
    return object_point_list, image_points_list
        


# In[6]:

object_point_list, image_points_list = find_inside_corners(image_file_names)


# # Calibrate using points

# In[7]:

image_original, image_gray = get_sample_gray(image_file_names[1])

# Returns:
# - camera matrix
# - distortion coefficients
# - rotation vectors
# - translation vectors
has_sucess, matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
    object_point_list, 
    image_points_list, 
    image_gray.shape[::-1], 
    None, 
    None)


# ## This instead of concatinating edges, stretch/curve them

# In[8]:

image_dimentions = image_original.shape[:2] # height, width
optimized_matrix, roi = cv2.getOptimalNewCameraMatrix(
    matrix, 
    distortion, 
    image_dimentions, 
    1, 
    image_dimentions)


# # Remove the distortion form the images

# In[9]:

for image_file_name in image_file_names:
    
    print("removing distortion in", image_file_name)
    image_original = cv2.imread(image_file_name)

    print("without edge distortion (cropping)")
    image1 = cv2.undistort(image_original, matrix, distortion, None, None)
    plot_images(image_original, image1)
    
    # save to disk
    cv2.imwrite("output_images/" + "unoptimized_" + image_file_name, image1)
    
    print("with edge distortion (curved)")
    image2 = cv2.undistort(image_original, matrix, distortion, None, optimized_matrix ) 
    plot_images(image_original, image2)
    
    # save to disk
    cv2.imwrite("output_images/" + "optimized_" + image_file_name, image2)


# # Conclusion
# 
# The **optimized** image provides a better camera calibration because it maintains more area of the image instead of cropping it. This is especially visible in the example below. I have manually drawn the **red lines** to show the straightness of on the **corrected image on the right**. The edge curvature plays mental/optical trics. 
# 
# I could improve even further if I took more chessboard samples. 
# 
# The final image might be cropped to hide the curved edges.
# 
# <img src="example_calibration.png" />

# In[ ]:




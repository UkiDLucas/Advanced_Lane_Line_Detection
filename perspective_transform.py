
# coding: utf-8

# In[1]:

def warp(image):
    print(image.shape) # height, width, channels
    image_size = (image.shape[1], image.shape[0]) # width, height
    
    src = np.float32(
    [[352,83],
     [354,116],
     [281,108],
     [280,74]
    ])
    
    dst = np.float32(
    [[352,82],
     [352,116],
     [280,116],
     [280,82]
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, image_size, flags=cv2.INTER_LINEAR)
    return 


# In[2]:

def corners_unwarp(image, nx, ny, camera_matrix, distortion_coefficients):
    """
    Function takes: 
    - an original image
    - chessboard dimantions nx, ny e.g. 9, 6
    - perspective_transform_matrix
    - distortion coefficients
    Returns:
    - new perspective-transformed image
    - perspective_transform_matrix
    """
    
    image_undist = cv2.undistort(
        image, 
        camera_matrix, 
        distortion_coefficients, 
        None, 
        perspective_transform_matrix)
    
    image_gray = cv2.cvtColor(image_undist, cv2.COLOR_BGR2GRAY)
    # Search for corners in the grayscaled image
    ret, corners = cv2.findChessboardCorners(image_gray, (nx, ny), None)

    if ret == True:
        # If we found corners, draw them! (just for fun)
        cv2.drawChessboardCorners(image_undist, (nx, ny), corners, ret)
        # Choose offset from image corners to plot detected corners
        # This should be chosen to present the result at the proper aspect ratio
        # My choice of 100 pixels is not exact, but close enough for our purpose here
        offset = 100 # offset for dst points
        # Grab the image shape
        img_size = (image_gray.shape[1], image_gray.shape[0])

        # For source points I'm grabbing the outer four detected corners
        src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
        # For destination points, I'm arbitrarily choosing some points to be
        # a nice fit for displaying our warped result 
        # again, not exact, but close enough for our purposes
        dst = np.float32([[offset, offset], [img_size[0]-offset, offset], 
                                     [img_size[0]-offset, img_size[1]-offset], 
                                     [offset, img_size[1]-offset]])
        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(image_undist, M, img_size)

    # Return the resulting image and matrix
    return warped, M


# In[ ]:




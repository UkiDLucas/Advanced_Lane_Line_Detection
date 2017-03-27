
# Camera calibration: adjusting camera lense edge distortion
by Uki D. Lucas

## Motivation

The raw camera data usually has a certain level of distortion caused by lense shape, this is especially pronounced on the edges of the image. The correction is essential in applications like image recognition where mainitaining the shape is essential, especially in autonomous vehicles, robotics and even in 3D printing.

## Solution approach

The common solution is to compare a **known shape object e.g. a chessboard** with the image taken, then calculate this **specific camera's** adjustment parameters that then can be applied to every frame taken by the camera. If the camera changes (camera lense type, resolution, image size) in the set up, the new chessboard images have to be taken and calibration process repeated.

The angle of chessboard in relation to the camera does not matter, in fact the sample pictures should be taken from various angles. Only the pictures with **whole chessboard showing** will work.

You should take **many, at least 20 sample images**, any less and the learning process renders images that are not a well corrected.


```python
# When working on this file set to True, 
# when using as library, or commiting set to False
should_run_tests_on_camera_calibration = False
```

# Desired result (spoiler alert!)

For people who have no patience to read the whole paper I am including the final result:


<img src="example_calibration.png" />


```python
import numpy as np # used for lists, matrixes, etc.
import cv2 # we will use OpenCV library
%matplotlib inline
```

## Useful helper method


```python
def plot_images(left_image, right_image):
    import numpy as np
    import matplotlib.pyplot as plt
    plt.figure(figsize=(20,10))
    plot_image = np.concatenate((left_image, right_image), axis=1)
    plt.imshow(plot_image)
    plt.show() 
```


```python
def __get_sample_gray(image_file_name: str):
    import cv2 # we will use OpenCV library
    image_original = cv2.imread(image_file_name)
    # convert BGR image to gray-scale
    image_gray = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
    return image_original, image_gray
    
```

## The pattern will look for "inner corners" e.g. black touching the black square


```python
def __find_inside_corners(image_file_names: list, nx: int=9, ny: int=6, verbose = False):
    # Chessboard dimentsions
    # nx = 9 # horizontal
    # ny = 6 # vertical

    import cv2 # we will use OpenCV library
    import numpy as np
    # Initialise arrays

    # Object Points: points on the original picture of chessboard
    object_point_list = []

    #Image Points: points on the perfect 2D chessboard
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
        
        if verbose:
            print("processing image:", image_file_name)
        image_original, image_gray = __get_sample_gray(image_file_name)

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

            if verbose:
                plot_images(image_original, image_corners)
        else: # not has_found
            if verbose:
                print("The", chessboard_dimentions, 
                  "chessboard pattern was not found, most likely partial chessboard showing")
                plt.figure()
                plt.imshow(image_original)
                plt.show()
        # end if has_found
    # end for
    return object_point_list, image_points_list
        
```

# Calibrate using points


```python
def prep_calibration(image_file_names: list, use_optimized = True, verbose = False):
    import cv2 # we will use OpenCV library
    # find CORNERS
    object_point_list, image_points_list = __find_inside_corners(image_file_names)
    
    # get smaple image, mostly for dimensions
    image_original, image_gray = __get_sample_gray(image_file_names[1])

    # Learn calibration
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
    
    ## I can use this to improve the calibration (no cropped edges, but curved edges)
    image_dimentions = image_original.shape[:2] # height, width
    matrix_optimized, roi = cv2.getOptimalNewCameraMatrix(
        matrix, 
        distortion, 
        image_dimentions, 
        1, 
        image_dimentions)
    return matrix, matrix_optimized, distortion
```


```python
def apply_correction(
    image_file_name: str = None, 
    matrix = None, 
    distortion = None,
    matix_optimized = None): # optional
    import cv2 # we will use OpenCV library
    
    print("Removing distortion in", image_file_name)
    image = cv2.imread(image_file_name)
    
    image_corrected = cv2.undistort(image, matrix, distortion, None, matix_optimized)
    
    if matix_optimized is None:
        print("NOT OPTIMIZED: edges are cropped.")
    else:
        print("OPTIMIZED: fuller image, but with edge distortion.")
        
    plot_images(image, image_corrected)
    return image_corrected
```

## Get list of sample chessboard images by this camera


```python
if should_run_tests_on_camera_calibration:
    # read a list of files using a parern
    import glob
    image_file_names = glob.glob("camera_cal/calibration*.jpg") # e.g. calibration19.jpg
    print("found", len(image_file_names), "calibration image samples" )
    print("example", image_file_names[0])
```


```python
#if should_run_tests_on_camera_calibration:
#    object_point_list, image_points_list = find_inside_corners(image_file_names)
```


```python
if should_run_tests_on_camera_calibration:
    matrix, matrix_optimized, distortion = prep_calibration(image_file_names, use_optimized = True)
```

# Remove the distortion form the images


```python
if should_run_tests_on_camera_calibration:
    image_file_name = "test_images/test1.jpg"
    image_corrected = apply_correction(image_file_name, matrix, distortion)
    image_corrected = apply_correction(image_file_name, matrix, distortion, matrix_optimized)
    # save to disk
    cv2.imwrite("output_images/" + "optimized_" + image_file_name, image_corrected)
```


```python
if should_run_tests_on_camera_calibration:
    for image_file_name in image_file_names:
        import cv2 # we will use OpenCV library

        image_corrected = apply_correction(image_file_name, matrix, distortion)
        image_corrected = apply_correction(image_file_name, matrix, distortion, matrix_optimized)

        # save to disk
        cv2.imwrite("output_images/" + "optimized_" + image_file_name, image_corrected)
```

# Conclusion

The **optimized** image provides a better camera calibration because it maintains more area of the image instead of cropping it. This is especially visible in the example below. I have manually drawn the **red lines** to show the straightness of on the **corrected image on the right**. The edge curvature plays mental/optical trics. 

I could improve even further if I took more chessboard samples. 

The final image might be cropped to hide the curved edges.

<img src="example_calibration.png" />


```python
# see http://nbconvert.readthedocs.io/en/latest/usage.html
!jupyter nbconvert --to markdown camera_calibration.ipynb
```

    [NbConvertApp] Converting notebook camera_calibration.ipynb to markdown
    [NbConvertApp] Writing 8918 bytes to camera_calibration.md


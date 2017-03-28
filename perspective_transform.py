
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


# In[ ]:




# In[ ]:




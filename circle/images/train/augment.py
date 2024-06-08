import random
import cv2
import os
import glob
import numpy as np
from PIL import Image

# Function to create a mosaic from input images and annotations
def mosaic(all_img_list, all_annos, idxs, output_size, scale_range, filter_scale=0):
    # Create an empty canvas for the output image
    output_img = np.zeros([output_size[0], output_size[1], 3], dtype=np.uint8)
    
    # Randomly select scales for dividing the output image
    scale_x = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])
    scale_y = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])
    
    # Calculate the dividing points based on the selected scales
    divid_point_x = int(scale_x * output_size[1])
    divid_point_y = int(scale_y * output_size[0])

    # Initialize a list for new annotations
    new_anno = []
    
    # Process each index and its respective image
    for i, idx in enumerate(idxs):
        path = all_img_list[idx]  # Image path
        img_annos = all_annos[idx]  # Image annotations

        img = cv2.imread(path)  # Read the image
        
        # Place each image in the appropriate quadrant of the output image
        if i == 0:  # top-left quadrant
            img = cv2.resize(img, (divid_point_x, divid_point_y))
            output_img[:divid_point_y, :divid_point_x, :] = img
            for bbox in img_annos:  # Update annotations accordingly
                xmin = bbox[1] - bbox[3]*0.5
                ymin = bbox[2] - bbox[4]*0.5
                xmax = bbox[1] + bbox[3]*0.5
                ymax = bbox[2] + bbox[4]*0.5

                xmin *= scale_x
                ymin *= scale_y
                xmax *= scale_x
                ymax *= scale_y
                new_anno.append([bbox[0], xmin, ymin, xmax, ymax])

        # Repeat the process for other quadrants (top-right, bottom-left, bottom-right)
        # Updating image placement and annotations accordingly
        
    # Filter annotations based on the provided scale
    if 0 < filter_scale:
        new_anno = [anno for anno in new_anno if
                    filter_scale < (anno[3] - anno[1]) and filter_scale < (anno[4] - anno[2])]

    return output_img, new_anno

# Example data (replace with your own data)
all_img_list = ['1.jpg', '2.jpg', '3.jpg', '4.jpg']
# List of image paths
all_annos = [
    [[0,0.588450,0.549890,0.223684,0.163377], [0, 0.539474,0.582785,0.178363,0.132675]],  # Annotations for image 1
    [[0,0.462993,0.600512,0.095943,0.130117], [0,0.533333,0.538580,0.082292,0.119599]],  # Annotations for image 2
    #... for other images
]

idxs = [0, 1, 2, 3]  # Indices representing images for the mosaic
output_size = (600, 600)  # Dimensions of the final mosaic image
scale_range = (0.7, 0.9)  # Range of scaling factors applied to the images
filter_scale = 20  # Optional filter for bounding box sizes

# Debugging - Print out values for inspection
print("Number of images:", len(all_img_list))
print("Number of annotations:", len(all_annos))
print("Indices for mosaic:", idxs)

# Call the mosaic function
mosaic_img, updated_annotations = mosaic(all_img_list, all_annos, idxs, \
output_size, scale_range, filter_scale)

# Display or use the generated mosaic_img and updated_annotations
# For instance, you can display the mosaic image using OpenCV
cv2.imshow('Mosaic Image', mosaic_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Access and use the updated_annotations for further processing
print("Updated Annotations:")
print(updated_annotations)

#!/usr/bin/env python3
import cv2
import numpy as np
import os

# Directory containing the images
image_dir = './input_images/'

# Output directory
output_dir = './output_images/'
os.makedirs(output_dir, exist_ok=True)

# Iterate over all images in the directory
for filename in os.listdir(image_dir):
    #print(filename)
    if filename == 'CEK AL 24hpi_0004.jpg' and filename.endswith(".jpg") or filename.endswith(".png"):  # Add/modify the file types based on your needs
        print(filename)
        image_path = os.path.join(image_dir, filename)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Enhance contrast if necessary using histogram equalization
        equalized_image = cv2.equalizeHist(gray_image)

        # Apply a binary threshold to get a binary image
        thresh = 50 # This is just an example value; adjust based on your image
        _, binary_image = cv2.threshold(equalized_image, thresh, 255, cv2.THRESH_BINARY_INV)

        # Apply morphological operations to separate conglomerated cells
        kernel = np.ones((3,3), np.uint8)
        separated = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=2)

        # Find contours which will be the individual cells
        contours, _ = cv2.findContours(separated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Perform connected components analysis
        num_labels, labels_im = cv2.connectedComponents(binary_image)

        # Filter out small spots
        min_area = 300  # Define your size threshold here
        large_spots_mask = np.zeros_like(binary_image, dtype=np.uint8)
        for label in range(1, num_labels):
                label_mask = (labels_im == label).astype(np.uint8)
                if cv2.countNonZero(label_mask) >= min_area:
                    large_spots_mask[labels_im == label] = 255

        # Count the large dark spots
        num_large_spots, _ = cv2.connectedComponents(large_spots_mask)
        num_large_spots -= 1  # Subtract one for the background label

        print(f'File: {filename}, Number of large dark spots: {num_large_spots}')

        # Count the cells
        cell_count = len(contours)

        print(f'File: {filename}, Cell count: {cell_count}')
        
        # Count the cells
        cell_count = len(contours)

        # Optionally draw contours on the image to visualize
        for cnt in contours:
            cv2.drawContours(image, [cnt], 0, (0,0,255), 2)

        # Count the large dark spots
        num_large_spots, labels_im = cv2.connectedComponents(large_spots_mask)
        num_large_spots -= 1  # Subtract one for the background label

        # Annotate the image with the counts
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, f'Large spots: {num_large_spots}', (10, 30), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f'Cells: {cell_count}', (10, 60), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Save the result in the output directory
        output_filename = os.path.join(output_dir, 'CPE_counted_' + filename)
        cv2.imwrite(output_filename, image)
        #break
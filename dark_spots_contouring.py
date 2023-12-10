#!/usr/bin/env python3
import cv2
import numpy as np
import os
from skimage.measure import regionprops
import pandas as pd
import glob

# Directory containing the images
input_dir = './input_images/'

# Output directory
output_dir = './output_images/'
os.makedirs(output_dir, exist_ok=True)

# Iterate over all images in the directory
for filename in os.listdir(input_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Add/modify the file types based on your needs
        print(filename)
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Enhance contrast if necessary using histogram equalization
        equalized_image = cv2.equalizeHist(gray_image)

        # Apply a binary threshold to get a binary image
        thresh = 50 # adjust based on your image
        _, binary_image = cv2.threshold(equalized_image, thresh, 255, cv2.THRESH_BINARY_INV)

        # Apply morphological operations to separate conglomerated cells
        kernel = np.ones((3,3), np.uint8)
        separated = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=2)

        # Perform connected components analysis
        num_labels, labels_im = cv2.connectedComponents(binary_image)

        # Filter out small spots
        min_area = 1000  # Define your size threshold here
        large_spots_mask = np.zeros_like(binary_image, dtype=np.uint8)
        for label in range(1, num_labels):
                label_mask = (labels_im == label).astype(np.uint8)
                if cv2.countNonZero(label_mask) >= min_area:
                    large_spots_mask[labels_im == label] = 255

        # Count the large dark spots
        num_large_spots, _ = cv2.connectedComponents(large_spots_mask)
        num_large_spots -= 1  # Subtract one for the background label

        print(f'File: {filename}, Number of large dark spots: {num_large_spots}')

        # # Find contours which will be the individual cells
        # contours, _ = cv2.findContours(separated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # # # Count the cells
        # # cell_count = len(contours)

        # print(f'File: {filename}, Cell count: {cell_count}')

        # # Optionally draw contours on the image to visualize
        # for cnt in contours:
        #     cv2.drawContours(image, [cnt], 0, (0,255,0), 2)

        # Perform connected components analysis
        num_large_spots, labels_im = cv2.connectedComponents(large_spots_mask)
        num_large_spots -= 1  # Subtract one for the background label

        # Find contours which will be the large dark spots
        contours, _ = cv2.findContours(large_spots_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Optionally draw contours on the image to visualize
        for cnt in contours:
            cv2.drawContours(image, [cnt], 0, (0,255,0), 2)

        # Annotate the image with the counts
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        # text_position = (10, min(30, image.shape[0] - 10))  # Ensure the text position is within the image
        text_position = (10, max(20, image.shape[0] // 10))  # Start the text 10% down from the top of the image
        text_color = (0, 255, 0)  # Set the text color to white for better visibility
        # Extract the base filename without the directory or extension
        base_filename = os.path.basename(filename)
        base_filename = os.path.splitext(base_filename)[0]

        # Annotate the image with the filename and the counts
        cv2.putText(image, f'{base_filename}: Large spots: {num_large_spots}', text_position, font, 10, text_color, 2, cv2.LINE_AA)
        # cv2.putText(image, f'Large spots: {num_large_spots}', text_position, font, 1, text_color, 2, cv2.LINE_AA)
        
        # Save the result in the output directory
        output_filename = os.path.join(output_dir, 'CPE_counted_' + filename)
        cv2.imwrite(output_filename, image)

        #Initialize a DataFrame to store the measurements
        df = pd.DataFrame(columns=['image', 'label', 'area', 'perimeter'])

        # Loop over all images in the directory
        for filename in glob.glob(os.path.join(input_dir, '*.jpg')):  # Adjust the pattern to match your images
            # Load the image
            image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

            # Apply your image processing steps here...

            # Perform connected components analysis
            num_large_spots, labels_im = cv2.connectedComponents(large_spots_mask)
            num_large_spots -= 1  # Subtract one for the background label

            # Calculate measurements for each large dark spot
            for label in range(1, num_large_spots + 1):
                # Create a mask for the current large dark spot
                spot_mask = (labels_im == label).astype(np.uint8)

                # Calculate area
                area = cv2.countNonZero(spot_mask)

                # Calculate perimeter
                contours, _ = cv2.findContours(spot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                perimeter = cv2.arcLength(contours[0], True)

                # Add the measurements to the DataFrame
                df = pd.concat([df, pd.DataFrame({'image': [filename], 'label': [label], 'area': [area], 'perimeter': [perimeter]})], ignore_index=True)
       
                # Save the measurements to a CSV file
                df.to_csv(os.path.join(output_dir, 'measurements.csv'), index=False)
                
        
import matplotlib.pyplot as plt
import seaborn as sns

# Load the measurements from the CSV file
df = pd.read_csv(os.path.join(output_dir, 'measurements.csv'))

# Create a scatter plot of area vs perimeter for each image
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='area', y='perimeter', hue='image')

# Set the plot title and labels
plt.title('Area vs Perimeter for each image')
plt.xlabel('Area (pixels²)')
plt.ylabel('Perimeter (pixels)')

# Show the plot
plt.show()

# Create a boxplot of total area for each image
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='image', y='area')



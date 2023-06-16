import os
import cv2
import numpy as np

# Define the source and destination directories
src_dir = r"C:\Users\dmmon\Desktop\Computer Science\year 3\Project Gmar\new_digit_data"
dst_dir = r"C:\Users\dmmon\Desktop\Computer Science\year 3\Project Gmar\Digit_Data_Binary_new"

# Create the destination directory if it doesn't exist
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

# Iterate over the subdirectories in the source directory
for subdir in os.listdir(src_dir):
    # Create a corresponding subdirectory in the destination directory
    if not os.path.exists(os.path.join(dst_dir, subdir)):
        os.makedirs(os.path.join(dst_dir, subdir))

    # Iterate over the files in the subdirectory
    for filename in os.listdir(os.path.join(src_dir, subdir)):
        # Read the image
        gray = cv2.imread(os.path.join(src_dir, subdir, filename), cv2.IMREAD_GRAYSCALE)
        plate_img_gray = cv2.equalizeHist(gray)  # Apply histogram equalization
        plate_img_gray = cv2.GaussianBlur(plate_img_gray, (3, 3), 0)  # Apply Gaussian blur
        _, plate_img_binary = cv2.threshold(plate_img_gray, 90, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((3, 3), np.uint8)
        plate_img_binary = cv2.dilate(plate_img_binary, kernel, iterations=1)
        plate_img_binary = cv2.erode(plate_img_binary, kernel, iterations=1)

        # Save the binary image in the destination directory
        base_filename = os.path.splitext(filename)[0]  # Get the filename without the extension
        cv2.imwrite(os.path.join(dst_dir, subdir, base_filename + '.jpg'), plate_img_binary)

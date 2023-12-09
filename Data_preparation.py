import os
import shutil
import nibabel as nib
import numpy as np
import cv2
import tensorflow as tf
import imageio.v2 as imageio

import matplotlib.pyplot as plt

# Allow GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to allocate only a specific amount of GPU memory
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            
        # Optional: Limit the amount of GPU memory TensorFlow can allocate
        # tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
        
    except RuntimeError as e:
        print(e)

source_folder = r"input_Directory_To_Folder"
mask_folder = os.path.join(source_folder, 'mask')
image_folder = os.path.join(source_folder, 'image')
image_mip_folder = os.path.join(source_folder, 'image_MIP')
mask_mip_folder=os.path.join(source_folder, 'mask_MIP')

# Creates necessary folders if they don't exist
for folder in [mask_folder, image_folder, image_mip_folder]:
    os.makedirs(folder, exist_ok=True)

for folder_name in os.listdir(source_folder):
    folder_path = os.path.join(source_folder, folder_name)
    nifti_files = [file for file in os.listdir(folder_path) if file.endswith('.nii') or file.endswith('.nii.gz')]

    for nifti_file in nifti_files:
        source_file_path = os.path.join(folder_path, nifti_file)
        file_base_name, file_extension = os.path.splitext(nifti_file)
        destination_file_path = os.path.join(image_folder if "_NCCT" in nifti_file else mask_folder, nifti_file)
        destination_mip_file_path = os.path.join(image_mip_folder, f"{file_base_name}_MIP.png")

        # Skips if the file already exists in the destination folders
        if not os.path.exists(destination_file_path):
            shutil.copy(source_file_path, destination_file_path)

def calculate_and_save_mip(input_folder, output_folder):
    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        if file_name.endswith('.nii.gz'):
            base_name = os.path.splitext(file_name)[0].split("_")[0]
            
            try:
                # Load NIfTI image
                nifti_image = nib.load(file_path)
                nifti_data = nifti_image.get_fdata()
                
                # Calculate Maximum Intensity Projection (MIP)
                mip = np.max(nifti_data, axis=2)
                
                # Normalize and convert to CV_8U
                mip_normalized = cv2.normalize(mip, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                
                # Save the MIP image
                output_filename = f"{base_name}_MIP.png"
                output_path = os.path.join(output_folder, output_filename)
                cv2.imwrite(output_path, mip_normalized)
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue

# Process images in image_folder and save MIP images in image_mip_folder
image_folder = r'input_Directory_To_Folder\image'
image_mip_folder = r'input_Directory_To_Folder\image_MIP'
calculate_and_save_mip(image_folder, image_mip_folder)

# Process masks in mask_folder and save MIP images in mask_mip_folder
mask_folder = r'input_Directory_To_Folder\mask'
mask_mip_folder = r'input_Directory_To_Folder\mask_MIP'
calculate_and_save_mip(mask_folder, mask_mip_folder)







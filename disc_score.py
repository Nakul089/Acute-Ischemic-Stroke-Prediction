import numpy as np
from PIL import Image

def dice_coefficient(pred_mask, sample_mask):
    intersection = np.sum(pred_mask * sample_mask)
    union = np.sum(pred_mask) + np.sum(sample_mask)
    dice = (2.0 * intersection) / (union + 1e-6)  # Adding a small constant to avoid division by zero
    return dice

# Load the generated mask and ground truth mask images
generated_mask_path = r'C:\Proxmed\predicted_mask.jpg' 
ground_truth_mask_path = r'C:\Proxmed\Dataset Proxmed\extracts\New folder\brain_mask\Anon1.jpg'  

generated_mask_image = Image.open(generated_mask_path)
generated_mask_image = generated_mask_image.resize((256, 256))

# Load and resize the ground truth mask image
ground_truth_mask_image = Image.open(ground_truth_mask_path)
ground_truth_mask_image = ground_truth_mask_image.resize((256, 256))

# Convert the images to NumPy arrays
generated_mask = np.array(generated_mask_image)
ground_truth_mask = np.array(ground_truth_mask_image)

# Convert to binary masks
threshold = 0.5
generated_mask_binary = (generated_mask > threshold).astype(np.uint8)
ground_truth_mask_binary = (ground_truth_mask > 0).astype(np.uint8)

# Calculate the Dice coefficient
dice_score = dice_coefficient(generated_mask_binary, ground_truth_mask_binary)

print(f"Dice Coefficient: {dice_score:.4f}")
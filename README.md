# Introduction
Development of  a robust and efficient algorithm or AI model capable of accurately segmenting the hypodense region from Brain Non-Contrast Computed Tomography (NCCT) images, regardless of the slice thickness and orientation of the images. 
#Data Preparation
To begin, we organize our dataset into two folders: 'images' containing masks of brain regions and 'mask' containing the corresponding NCCT (Non-Contrast Computed Tomography ) images.
#Preprocessing
Each brain mask and CTA image undergoes preprocessing before being used for training. The process involves generating Maximum Intensity Projections (MIPs) along the z-axis to capture important features, followed by normalization and decoding a PNG image into a TensorFlow tensor and then attempts to find the maximum value within that tensor and resizing of the image. The results are saved as PNG files in dedicated folders.
#U-Net Model
The U-Net architecture is employed due to its effectiveness in image segmentation tasks. It consists of an encoder path to capture context and a decoder path for precise localization.

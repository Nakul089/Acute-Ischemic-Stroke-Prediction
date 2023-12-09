import os
import matplotlib.pyplot as plt
import nibabel as nib
import tensorflow as tf

path = r'input_Directory_To_Folder'
image_path = os.path.join(path, 'image')
mask_path = os.path.join(path, 'mask')
image_list = os.listdir(image_path)
mask_list = os.listdir(mask_path)
image_list = [os.path.join(image_path, i) for i in image_list]
mask_list = [os.path.join(mask_path, i) for i in mask_list]

#To check out some of the unmasked and masked images from the dataset:
for im_path, mask_path in zip(image_list, mask_list):
    nifti_img = nib.load(im_path)
    nifti_mask = nib.load(mask_path)
    
    data_img = nifti_img.get_fdata()
    data_mask = nifti_mask.get_fdata()
    
    slice_to_view = data_img.shape[2] // 2  # z-axis as slices


    # Displaying the image slice
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(data_img[:, :, slice_to_view], cmap='gray')
    plt.title('Image Slice')
    plt.axis('off')
    
    # Displaying the mask slice
    plt.subplot(1, 2, 2)
    plt.imshow(data_mask[:, :, slice_to_view], cmap='viridis')  # Adjust cmap as needed
    plt.title('Mask Slice')
    plt.axis('off')
    
    plt.show()
    plt.close() 

#Spliting Dataset into Unmasked and Masked Images  
image_list_ds = tf.data.Dataset.list_files(image_list, shuffle=False)
mask_list_ds = tf.data.Dataset.list_files(mask_list, shuffle=False)

for path in zip(image_list_ds.take(3), mask_list_ds.take(3)):
    print(path)

image_filenames = tf.constant(image_list)
masks_filenames = tf.constant(mask_list)

dataset = tf.data.Dataset.from_tensor_slices((image_filenames, masks_filenames))

for image, mask in dataset.take(1):
    print(image)
    print(mask)

#Preprocessing the  Data
def process_path(image_path, mask_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=3)
    mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)
    return img, mask

def preprocess(image, mask):
    input_image = tf.image.resize(image, (96, 128), method='nearest')
    input_mask = tf.image.resize(mask, (96, 128), method='nearest')

    input_image = input_image / 255.

    return input_image, input_mask

image_ds = dataset.map(process_path)
processed_image_ds = image_ds.map(preprocess)


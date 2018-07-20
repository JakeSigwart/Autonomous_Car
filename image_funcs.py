import numpy as np
import matplotlib.pyplot as plt
import picamera
import picamera.array
import time
from PIL import Image
import io


#Input: int data array representing an image        Output: image on screen
def plot_image(img_data):
    plt.axis('off')
    plt.imshow(img_data)
    plt.show()

#Input: file path to .png image file.   Output: numpy array
def png_to_array(file_path):
    img = Image.open(file_path)
    arr = np.array(img)
    return arr

#Display camera view on screen for 'time_sec' seconds
def camera_preview(time_sec):
    camera = picamera.PiCamera()
    camera.resolution = (2592,1944)
    camera.start_preview()
    time.sleep(time_sec)
    camera.stop_preview()


#Input: array of unsigned int images
#Output: array of float images ready for machine learning
def normalize_images(images, span=1.0, min_val=0.0):
    images_out = span*(np.array(images, dtype=np.float32) / 255.0)  + min_val
    return images_out

#Return images to uint values from 0 to 255
def denormalize_images(images, span=1.0, min_val=0.0):
    images_out = np.array(np.array(images - min_val)*255.0 / span,  dtype=np.uint8)
    return images_out


#Input: int data array representing an image        Output: image on screen
def plot_image(img_data):
    plt.axis('off')
    plt.imshow(img_data)
    plt.show()

#Input: file path to .png image file.   Output: numpy array
def png_to_array(file_path):
    img = Image.open(file_path)
    arr = np.array(img)
    return arr

#Display camera view on screen for 'time_sec' seconds
def camera_preview(time_sec):
    camera = picamera.PiCamera()
    camera.resolution = (2592,1944)
    camera.start_preview()
    time.sleep(time_sec)
    camera.stop_preview()

#Compute mean values for regions of image
def image_get_mean_rgbs(image, fsize=[8,8]):
    nrows = image.shape[0]
    ncols = image.shape[1]
    output = np.zeros([int(nrows/fsize[0]), int(ncols/fsize[1]), 3], dtype=np.uint8)
    
    for rowc in range(0, int(nrows/fsize[0])):
        for colc in range(0, int(ncols/fsize[1])):
            block = image[rowc*fsize[0]:(rowc+1)*fsize[0], colc*fsize[1]:(colc+1)*fsize[1]]
            block_mean = np.mean(np.mean(block, axis=0), axis=0)
            block_mean = block_mean.astype(np.uint8)
            output[rowc,colc] = block_mean
    return output
            
def image_get_mean_brightnesses(image, fsize=[8,8]):
    a = image_get_mean_rgbs(image, fsize=fsize)
    output = np.mean(a/255.0, axis=2)
    return output


    
            
            
            
            
    
    
    

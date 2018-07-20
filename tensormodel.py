import tensorflow as tf

#Functions to be utalized in Tensorflow models

#Input: an image, the size of the crop square and number of channels
#Output: Cropped image
def crop_image(image, img_size_cropped, num_channels):
    image = tf.random_crop(image, size=[img_size_cropped, img_size_cropped, num_channels])
    return image

#Flips an image about the diagonal (top-left to bottom-right) axis
def Image_transpose(image):
    image = tf.image.transpose_image(image)
    return image

#Input: A grayscale image w/ dimensions: [height, width, 1]
#Output: An RGB image w/ dimensions: [height, width, 3]
def Image_grayscale_to_rgb(image):
    image = tf.image.grayscale_to_rgb(image, name=None)
    return image

#Input: An RGB image w/ dimensions: [height, width, 3]
#Output: A grayscale image w/ dimensions: [height, width, 1]
def Image_rgb_to_grayscale(image):
    image = tf.image.rgb_to_grayscale(image)
    return image


#Image Pre-processing
#Input: an image, a scalar int representing the number of rotations CCW by 90 degrees
#Output: The image rotated
def Image_rotate(image, num_rot):
    image = tf.image.rot90(image, k=1, name=None)
    return image

#Input: A single RGB image: [height, width, 3]
#Output: The RGB image with randomly adjusted hue, contrast, brightness, saturation
def Image_random_rgb_distort(image):
    image = tf.image.random_hue(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.7, upper=1.3)
    image = tf.image.random_brightness(image, max_delta=.2)
    image = tf.image.random_saturation(image, lower=0.7, upper=1.3)
    #Make sure color levels remain in the range [-0.5, 0.5]
    image = tf.minimum(image, -0.5)
    image = tf.maximum(image, 0.5)
    return image


#Randomly flip and adjust each of the input 
def pre_process_images(images, process_images):
    def f1(): return images
    def f2(): return tf.map_fn(lambda image: Image_random_rgb_distort(image), images)
    images = tf.cond(process_images, f2, f1)
    return images



  
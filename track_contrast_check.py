import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from Data_log_manager import *
from encode_funcs import *
from tensormodel import *
from image_funcs import *

num_of_file = 109

image_cols = 128
image_rows = 64
image_channels = 3
sonar_range = 500
steering_range = [223, 372, 521]
throttle_range = [226, 375, 524]
num_steering_classes = 31
num_throttle_classes = 31
steering_range_out = [223, 372, 521]
throttle_range_out = [300, 375, 450]

data_batch_size = 32

data_path = '/home/pi/Desktop/Autonomous_car/data/'
reader = Data_log_manager(data_path=data_path, batch_size=data_batch_size, num_files=1000)

delta_t, steering, throttle, sonar, accel, images = reader.load_data_files(start_file_num=num_of_file ,num_toread=1)

print(steering)
plot_image(images[0])

'''
means_array = image_get_mean_rgbs(images[0], fsize=[8,16])
print("Brightness magnitudes (out of 1.0):")
print(image_get_mean_brightnesses(images[0], fsize=[16,16]))
'''


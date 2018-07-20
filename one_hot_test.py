import numpy as np
from Data_log_manager import *
from encode_funcs import *
from image_funcs import *

#Test opening and reading a data file. Test one-hot encoding and decoding of data
steering_range = [223, 372, 521]
throttle_range = [226, 375, 524]
num_steering_classes = 31
num_throttle_classes = 31
steering_range_out = [223, 372, 521]
throttle_range_out = [300, 375, 450]

data_batch_size = 32
data_path = '/home/pi/Desktop/Autonomous_car/data/'


reader = Data_log_manager(data_path=data_path, batch_size=data_batch_size, num_files=1000)
delta_t, steering, throttle, sonar, accel, images = reader.load_data_files(start_file_num=0 ,num_toread=1)

#Format inputs, run tensorflow, format outputs
hot_steering = one_hot_encode(steering, steering_range, num_steering_classes)
hot_throttle = one_hot_encode(throttle, throttle_range, num_throttle_classes)
images_norm = normalize_images(images, span=1.0, min_val=-0.5)

steering_out = one_hot_decode(hot_steering, steering_range_out, num_steering_classes)
throttle_out = one_hot_decode(hot_throttle, throttle_range_out, num_throttle_classes)

plot_image(images[0])
print(images_norm[0])


import io
import os
import re
import numpy as np
import datetime as dt
import time
import serial
import tensorflow as tf
import Adafruit_PCA9685
import picamera
import picamera.array
from PIL import Image
import RPi.GPIO as GPIO

from Fast_camera import *
from Data_log_manager import *
from Raw_data_manager import *
from encode_funcs import *
from tensormodel import *

steering_pin = 13       #Hardware Configuration on PCA-9685 motor control board
throttle_pin = 14

mode_button_pin = 18        #Maintained button: Off-manual, On-autonomous
log_button_pin = 23         #Maintained button: Off-don't log data, On- log data
endrun_button_pin = 24      #Momentary button: push to immediately end run

arduino_port = '/dev/ttyACM0'
arduino_baud = 115200

image_cols = 128
image_rows = 64
image_channels = 3
sonar_range = 500     #cm
steering_range = [223, 372, 521] #full-Right, straight, full-left
throttle_range = [226, 375, 524] #max-reverse, stop, max-forward
num_steering_classes = 31
num_throttle_classes = 31
steering_range_out = [310, 372, 440]
throttle_range_out = [310, 375, 440]

data_batch_size = 32    #About 3 sec
max_num_data = 2400     #for about 2hr

data_path = '/home/pi/Desktop/Autonomous_car/data/'
event_path = '/home/pi/Desktop/Autonomous_car/data/logs/log.txt'
model_path = '/home/pi/Desktop/Autonomous_car/model/model.ckpt'
checkpt_path = '/home/pi/Desktop/Autonomous_car/model/checkpoint'

#Configure inputs
input_reader = Raw_data_manager(event_path)
input_reader.config_buttons(mode_pin=mode_button_pin, log_pin=log_button_pin, endrun_pin=endrun_button_pin)
input_reader.config_arduino(arduino_port=arduino_port, arduino_baud=arduino_baud, sonar_range=sonar_range)
input_reader.config_control_ranges(steering_range, throttle_range)
cam = Fast_camera(res_cols=image_cols, res_rows=image_rows, framerate=25, frame_batch_size=1, flip_vert=True)

#Configure outputs
data_log = Data_log_manager(data_path, batch_size=data_batch_size, num_files=max_num_data)
pwm = Adafruit_PCA9685.PCA9685()
pwm.set_pwm_freq(60)


#TENSORFLOW MODEL
graph = tf.Graph()
with graph.as_default():
    Training_status = tf.placeholder(tf.bool)
    Images = tf.placeholder(tf.float32, shape=[None, 64, 128, 3], name='Images')
    Accel = tf.placeholder(tf.float32, shape=[None, 3], name='Accel')
    Sonar = tf.placeholder(tf.float32, shape=[None], name='Sonar_normal')
    Steering_label = tf.placeholder(tf.float32, shape=[None, num_steering_classes], name='Steering_label')
    Throttle_label = tf.placeholder(tf.float32, shape=[None, num_throttle_classes], name='Throttle_label')
    
    proc_images = pre_process_images(Images, Training_status)
    
    W_conv1 = tf.Variable(tf.truncated_normal([6,6,3,32],mean=0.0,stddev=0.14), name='W_conv1')
    b_conv1 = tf.Variable(tf.constant(0.005, shape=[32]), name='b_conv1')
    h_conv1 = tf.nn.conv2d(proc_images, W_conv1, strides=[1,2,2,1], padding='SAME', name='h_conv1') + b_conv1
    h_pool1 = tf.nn.max_pool(tf.nn.relu(h_conv1), ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='h_pool1')
    #Size = [None, 16, 32, 32]    
    
    W_conv2 = tf.Variable(tf.truncated_normal([4,4,32,32],mean=0.0,stddev=0.0625), name='W_conv2')
    b_conv2 = tf.Variable(tf.constant(0.005, shape=[32]), name='b_conv2')
    h_conv2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[1,1,1,1], padding='SAME', name='h_conv2') + b_conv2
    h_pool2 = tf.nn.max_pool(tf.nn.relu(h_conv2), ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='h_pool2')
    #Size: [None, 8, 16, 32]
    
    h_reshaped = tf.reshape(h_pool2, [-1, 8*16*32])
    W_fc = tf.Variable(tf.truncated_normal([8*16*32, 64], mean=0.0, stddev=0.022))
    b_fc = tf.Variable(tf.constant(0.005, shape=[64]))
    h_fc = tf.nn.relu(tf.matmul(h_reshaped, W_fc)+b_fc)
    #Output shape: [None, 63]
    
    def train_keep(): return tf.constant(0.5, tf.float32)
    def run_keep(): return tf.constant(1.0, tf.float32)
    h_fc_dropout = tf.nn.dropout(h_fc, tf.cond(Training_status, train_keep, run_keep))
    
    W_readout_steering = tf.Variable(tf.truncated_normal([64,num_steering_classes], mean=0.0, stddev=0.18))
    W_readout_throttle = tf.Variable(tf.truncated_normal([64,num_throttle_classes], mean=0.0, stddev=0.18))
    b_readout_steering = tf.Variable(tf.constant(0.005, shape=[num_steering_classes]))
    b_readout_throttle = tf.Variable(tf.constant(0.005, shape=[num_throttle_classes]))
        
    hot_steering_opt_var = tf.matmul(h_fc_dropout, W_readout_steering)+b_readout_steering
    hot_throttle_opt_var = tf.matmul(h_fc_dropout, W_readout_throttle)+b_readout_throttle
    cross_entropy_steering = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Steering_label, logits=hot_steering_opt_var))
    cross_entropy_throttle = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Throttle_label, logits=hot_throttle_opt_var))

    opt_st = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_steering)
    opt_th = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_throttle)
    
    hot_steering_prediction = tf.nn.softmax(hot_steering_opt_var)
    hot_throttle_prediction = tf.nn.softmax(hot_throttle_opt_var)
    
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    
  
#TENSORFLOW SESSION
with tf.Session(graph=graph) as sess:
    sess.run(init)
    if os.path.isfile(checkpt_path):
        saver.restore(sess, model_path)
        print('Model restored from file.')
    else:
        print('Building new model...')
    start_up = True
    
    #Main Loop
    while start_up or (not endrun_state):
        
        #Get data, determine mode. Fix bad control values
        input_reader.update()
        
        timestamp, delta_t, steering, throttle, sonar, accel = input_reader.get_data()
        auto_state, log_state, endrun_state = input_reader.get_modes()
        
        #Display raw data for debugging purposes. Note: Displays raw throttle values not reduced values
        #input_reader.display_raw_data()
        #input_reader.display_mode_change()
        input_reader.update_event_log()

        data_array = [delta_t, steering, throttle, sonar, accel]
        
        #Conditionally take image. Save data.
        if log_state or auto_state:
            image = cam.capture_to_array()
        
        #Only log data if commands are received
        ignore_data = (steering<=steering_range[1]+3 and steering>=steering_range[1]-3) and (throttle<=throttle_range[1]+3 and throttle>=throttle_range[1]-3)
        if log_state and not ignore_data:
            data_log.update_data_log(data_array, image)
            input_reader.display_raw_data()
        
        #Autonomous
        if auto_state:            
            #Format inputs, run tensorflow, format outputs
            images_norm = normalize_images(image, span=1.0, min_val=-0.5)
            sonar_norm = sonar/sonar_range
            #delta_t
            accel = np.reshape(accel, (1,3))
            hot_steering_out, hot_throttle_out = sess.run([hot_steering_prediction, hot_throttle_prediction], feed_dict={Training_status:False, Images:images_norm} )
            print(hot_steering_out)
            
            #Reduce One-Hot vectors
            steering_out = one_hot_decode(hot_steering_out, steering_range_out, num_steering_classes)
            throttle_out = one_hot_decode(hot_throttle_out, throttle_range_out, num_throttle_classes)
            print('Auto S: '+str(steering_out)+'  T: ' +str(throttle_out))
            pwm.set_pwm(steering_pin, 0, steering_out)
            pwm.set_pwm(throttle_pin, 0, throttle_out)
            
        #Manual
        else:
            hot_steering_out = one_hot_encode(steering, steering_range, num_steering_classes)
            hot_throttle_out = one_hot_encode(throttle, throttle_range, num_throttle_classes)
            
            steering_out = one_hot_decode(hot_steering_out, steering_range_out, num_steering_classes)
            throttle_out = one_hot_decode(hot_throttle_out, throttle_range_out, num_throttle_classes)            
            #Reduced, discretized throttle values passed to motors
            
            pwm.set_pwm(steering_pin, 0, steering_out)
            pwm.set_pwm(throttle_pin, 0, throttle_out)
        start_up = False
    #End of main loop

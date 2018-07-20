import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from Data_log_manager import *
from encode_funcs import *
from tensormodel import *

start_batch = 20
num_batches = 20
save_run = True
training_status = True

image_cols = 128
image_rows = 64
image_channels = 3
sonar_range = 500
steering_range = [223, 372, 521]
throttle_range = [226, 375, 524]
num_steering_classes = 31
num_throttle_classes = 31
steering_range_out = [310, 372, 440]
throttle_range_out = [310, 375, 440]

data_batch_size = 32

data_path = '/home/pi/Desktop/Autonomous_car/data/'
model_path = '/home/pi/Desktop/Autonomous_car/model/model.ckpt'
checkpt_path = '/home/pi/Desktop/Autonomous_car/model/checkpoint'
log_path = '/home/pi/Desktop/Autonomous_car/logs/log.npy'


#Tensorflow model##########################
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
    
#Training session
with tf.Session(graph=graph) as sess:
    sess.run(init)
    if os.path.isfile(checkpt_path):
        saver.restore(sess, model_path)
        print('Model restored from file.')
    else:
        print('Building new model...')
    if os.path.isfile(log_path):
        old_entropy = np.load(log_path)
        entropy_track = np.zeros([num_batches], dtype=np.float32)
        old_len = old_entropy.shape[0]
    else:
        entropy_track = np.zeros([num_batches], dtype=np.float32)
        old_len = 0
        
  
    reader = Data_log_manager(data_path=data_path, batch_size=data_batch_size, num_files=2400)
    
    entropy_track = np.zeros([num_batches], dtype=np.float32)
                             
    for batch_n in range(0, num_batches):
        delta_t, steering, throttle, sonar, accel, images = reader.load_data_files(start_file_num=(start_batch + batch_n),num_toread=1)

        #Format inputs, run tensorflow, format outputs
        hot_steering = one_hot_encode(steering, steering_range, num_steering_classes)
        hot_throttle = one_hot_encode(throttle, throttle_range, num_throttle_classes)
        images_norm = normalize_images(images, span=1.0, min_val=-0.5)
        sonar_norm = sonar/sonar_range
        #Ignore: delta_t, accel
        hot_steering_out, hot_throttle_out, ce_st, ce_th, _, _ = sess.run([hot_steering_prediction, hot_throttle_prediction, cross_entropy_steering, cross_entropy_throttle, opt_st, opt_th], feed_dict={Training_status:training_status, Images:images_norm, Accel:accel, Sonar:sonar_norm, Steering_label:hot_steering, Throttle_label:hot_throttle} )
        
        #Actual values given to motors
        actual_st = one_hot_decode(hot_steering, steering_range_out)
        actual_th = one_hot_decode(hot_throttle, throttle_range_out)
        
        #Predictions
        pred_st = one_hot_decode(hot_steering_out, steering_range_out)
        pred_th = one_hot_decode(hot_throttle_out, throttle_range_out)
        
        entropy_track[batch_n] = ce_th + ce_st
        if training_status:
            print('Batch: '+str(batch_n)+' Cross entropy: '+str(ce_th+ce_st))
        else:
            st_acc = np.sum(actual_st == pred_st)/data_batch_size
            th_acc = np.sum(actual_th == pred_th)/data_batch_size
            print('Accuracy: S: '+str(st_acc)+' T: '+str(th_acc))
    
    if save_run and training_status:
        print("Saving run...")
        saver.save(sess, model_path)
        if old_len!=0:
            entropy_track = np.concatenate((old_entropy, entropy_track), axis=0)
        np.save(log_path, entropy_track)
    if training_status:
        plt.plot(entropy_track)
        plt.xlabel('Batches')
        plt.ylabel('Cross-Entropy')
        plt.show()
    
    
    
    
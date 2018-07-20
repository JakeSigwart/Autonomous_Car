import io
import os
import pickle
import numpy as np
import datetime as dt

#Purpose:   To batch data as it is received
#           Periodically save batch to file
class Data_log_manager:
    #Input: data file path,  num of entries per file,  stop saving after num_files
    def __init__(self, data_path, batch_size=100, num_files=1000):
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_files = num_files
        
        #Set batch counter to number of last file + 1
        self.batch_c = 0
        for n in range(0, num_files):
            if os.path.isfile(data_path+'batch_'+str(self.batch_c)+'.pickle'):
                self.batch_c = self.batch_c + 1
        
        self.elem_c = 0
        self.data = []
        
    #Purpose: Package data into array. Save array to file at fixed interval
    #Input: array of data, numpy array image
    #Output: to file, array: [[delta_t, steering_embedded, throttle_embedded, sonar, numpy accel, numpy image], [...], ... ]
    def update_data_log(self, data_array, img):
        data_array.append(img)
        self.data.append(data_array)
        self.elem_c = self.elem_c + 1
        if self.elem_c == self.batch_size:
            #Save data together
            with open(self.data_path+'batch_'+str(self.batch_c)+'.pickle', 'wb') as g:
                pickle.dump(self.data, g)
            #Save images as numpy array
            self.elem_c = 0
            self.batch_c = self.batch_c + 1
            self.data = []


    #Input: number of first file,  number of files to read 
    #Output: the data merged into arrays
    def load_data_files(self, start_file_num=0 ,num_toread=1):
        file_num = start_file_num
        data_whole = []
        start_up = True
        for n in range(0, num_toread):
            if os.path.isfile(self.data_path+'batch_'+str(file_num)+'.pickle'):
                file = open(self.data_path+'batch_'+str(file_num)+'.pickle', 'rb')
                data_array = pickle.load(file)
                file.close()
                if start_up:
                    for row in data_array:
                        data_whole.append(row)
                    start_up = False
                else:
                    data_whole.extend(data_array)
            else:
                break
            file_num = file_num + 1
        #Organize data before returning.
        num_rows = len(data_whole)
        times = []
        steerings = []
        throttles = []
        sonars = []
        accels = []
        images_ar = []
        for row in data_whole:
            times.append(row[0])
            steerings.append(row[1])
            throttles.append(row[2])
            sonars.append(row[3])
            accels.append(row[4])
            images_ar.append(row[5])
        delta_t = np.array(times, dtype=np.float32)
        steering = np.array(steerings, dtype=np.int32)
        throttle = np.array(throttles, dtype=np.int32)
        sonar = np.array(sonars, dtype=np.float32)
        accel = np.array(accels, dtype=np.float32)
        images = np.array(images_ar, dtype=np.uint8)                
        return delta_t, steering, throttle, sonar, accel, images
        
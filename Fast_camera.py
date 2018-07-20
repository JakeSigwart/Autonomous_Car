import picamera
import picamera.array
import time
import numpy as np
from PIL import Image
import io

#Class which takes images through the video port
#Performance:   flip_vert=True, @25fps: 0.061 sec/image
#               25fps appears to be the fastest for this class

class Fast_camera:
    #Input: num columns, num rows,   frame-rate to set video port,    num of frames read from the video port at a time
    #       Option to flip each image array
    def __init__(self, res_cols=128, res_rows=64, framerate=25, frame_batch_size=1, flip_vert=False):
        self.camera = picamera.PiCamera()
        self.camera.resolution = (res_cols, res_rows)
        self.camera.framerate = framerate
        self.frame_batch_size = frame_batch_size
        self.flip_vert = flip_vert
    
    #Helper function to manage data buffer
    def output_buff(self):
        stream = io.BytesIO()
        for i in range(0, self.frame_batch_size):
            yield stream
            stream.seek(0)
            img = Image.open(stream)
            img_array = np.array(img)   #Already uint8
            if i==0 and self.frame_batch_size==1:
                self.img_buffer = img_array
            if i==0 and self.frame_batch_size>1:
                self.img_buffer = np.reshape(img_array, (1,img_array.shape[0],img_array.shape[1],img_array.shape[2]))
            if i>0:
                self.img_buffer = np.concatenate((self.img_buffer, np.reshape(img_array, (1,img_array.shape[0],img_array.shape[1],img_array.shape[2]))), axis=0)
            stream.seek(0)
            stream.truncate()
        
    def output_file(self, file_path):
        stream = io.BytesIO()
        for i in range(0, self.frame_batch_size):
            yield stream
            #Do processing on image
            stream.seek(0)
            img = Image.open(stream)
            img_array = np.array(img)   #Already uint8
            np.save(file_path+str(i)+'.npy', img_array)
            stream.seek(0)
            stream.truncate()
    
    
    #Output:    Numpy array of dtype=uint8. If frame_batch_size=1: shape=[cols,rows,3].
    #           If frame_batch_size>1: shape=[frame_batch_size,cols,rows,3].
    def capture_to_array(self):
        self.camera.capture_sequence(self.output_buff(), 'jpeg', use_video_port=True)
        if self.flip_vert and self.frame_batch_size==1:
            self.img_buffer = np.rot90(self.img_buffer, k=2, axes=(0,1))
        if self.flip_vert and self.frame_batch_size>1:
            self.img_buffer = np.rot90(self.img_buffer, k=2, axes=(1,2))
        return self.img_buffer

    #Input: file path with beggining of file name. (for frame_batch_size>1: files will be numbered 0,1,2...)
    def capture_to_file(self, file_path):
        self.camera.capture_sequence(output_file(self, file_path), 'jpeg', use_video_port=True)        

        
        
        
        
        
        
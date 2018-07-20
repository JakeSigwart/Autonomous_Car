import io
import os
import re
import numpy as np
import time
import serial
import datetime as dt
import RPi.GPIO as GPIO

#Purpose:   Collect arduino data and button states.
#           Determine mode. Add mode changes to event log
class Raw_data_manager:
    #Input: file path to write events log to
    def __init__(self, event_path):
        self.event_path = event_path
        with open(self.event_path, 'a') as file:
            file.write('\n'+str(dt.datetime.now())+': Run started\n')
        self.last_mode_state = False
        self.last_log_state = False
        self.last_endrun_state = False
        self.start_up = True
        
    #Inputs: button pins
    def config_buttons(self, mode_pin=18, log_pin=23, endrun_pin=24):
        self.mode_pin = mode_pin
        self.log_pin = log_pin
        self.endrun_pin = endrun_pin
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(mode_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(log_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(endrun_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        #True state for self.last_mode_state indicates button is in the 'in' position
        #the 'in' position will mean autonomous mode is selected
        self.last_mode_state = GPIO.input(mode_pin)==False
        self.last_log_state = GPIO.input(log_pin)==False
        self.endrun_state = GPIO.input(endrun_pin)==False
    
    #Inputs: arduino settings
    def config_arduino(self, arduino_port='/dev/ttyACM0', arduino_baud=115200, sonar_range=500):
        self.arduino_port = arduino_port
        self.arduino_baud = arduino_baud
        self.ser = serial.Serial(self.arduino_port, self.arduino_baud)
        self.sonar_range = sonar_range
        self.start_up = True
        self.last_time = dt.datetime.now()
        
    def config_control_ranges(self, steering_range, throttle_range):
        self.steering_range = steering_range
        self.throttle_range = throttle_range
        
    #Read latest data in serial buffer
    def read_buffer_latest(self):
        data_str = ''
        while self.ser.inWaiting() > 0:
            data_line = self.ser.readline()
            data_str  = str(data_line)
        return data_str
    
    #Get the most recent data from arduino.
    #   Check button states and data to determine manual/autonomous mode.
    def update(self):
        #Read button states    
        mode_button_state = GPIO.input(self.mode_pin)==False
        self.log_state = GPIO.input(self.log_pin)==False
        self.endrun_state = GPIO.input(self.endrun_pin)==False
        #Read most recent full data-line from serial port
        data_str = self.read_buffer_latest()
        data = re.findall(r"[-+]?\d*\.\d+|\d+", data_str)
        while len(data)!=6:
            data_str = self.read_buffer_latest()
            data = re.findall(r"[-+]?\d*\.\d+|\d+", data_str)
        self.timestamp = dt.datetime.now()
        self.delta_t = ( self.timestamp - self.last_time ).total_seconds()
        self.last_time = self.timestamp
        self.steering = int(data[0])
        self.throttle = int(data[1])
        self.sonar = int(data[2])
        self.accel = np.array([float(data[3]), float(data[4]), float(data[5])], dtype=np.float32)
        #Fix bad values. If sonar==0: correct to 1 or sonar range
        if self.start_up:
            if self.sonar==0:
                self.sonar = self.sonar_range
            self.last_sonar = self.sonar
            self.start_up = False
        else:
            if self.sonar==0 and self.last_sonar>=self.sonar_range/2:
                self.sonar = self.sonar_range
            if self.sonar==0 and self.last_sonar<=int(self.sonar_range/2):
                self.sonar = 1
            self.last_sonar = self.sonar
        
        #Use bad steering values to determine whether control signal exists
        self.mode_state = mode_button_state and (self.steering==0 or self.throttle==0)
        
        #fix bad control values AFTER modes are determined
        if self.steering==0 or self.steering<=self.steering_range[0]-10 or self.steering>=self.steering_range[2]+10:
            self.steering = self.steering_range[1]
        if self.throttle==0 or self.throttle<=self.throttle_range[0]-10 or self.throttle>=self.throttle_range[2]+10:
            self.throttle = self.throttle_range[1]
        
        

    #Output data to main program
    def get_data(self):
        return self.timestamp, self.delta_t, self.steering, self.throttle, self.sonar, self.accel
    
    #Output the modes to main program
    def get_modes(self):
        return self.mode_state, self.log_state, self.endrun_state

    #Display data for debugging purposes
    def display_raw_data(self):
        print(str(self.steering)+' '+str(self.throttle)+' '+str(self.delta_t)+' '+str(self.sonar)+' '+str(self.accel))
    
    #Display modes for debugging purposes
    def display_mode_change(self):
        if (self.mode_state==True) and (self.last_mode_state==False): #Manual to autonomous
            print('Mode changed from Manual to Auto\n')
        if (self.mode_state==False) and (self.last_mode_state==True): #autonomous to manual
            print('Mode changed from Auto to Manual\n')  
        if (self.log_state==True) and (self.last_log_state==False): #Open files, Begin logging data
                print('Data logging started\n')
        if (self.log_state==False) and (self.last_log_state==True): #Stop logging data
            print('Data logging terminated\n')
        if self.endrun_state:
            print('Run ended\n')
        
    #Update the event log file
    def update_event_log(self):
        if (self.mode_state==True) and (self.last_mode_state==False): #Manual to autonomous
            with open(self.event_path, 'a') as file:
                file.write(str(dt.datetime.now())+': Mode changed from Manual to Auto\n')
        if (self.mode_state==False) and (self.last_mode_state==True): #autonomous to manual
            with open(self.event_path, 'a') as file:
                file.write(str(dt.datetime.now())+': Mode changed from Auto to Manual\n')            
        if (self.log_state==True) and (self.last_log_state==False): #Open files, Begin logging data
            with open(self.event_path, 'a') as file:
                file.write(str(dt.datetime.now())+': Data logging started\n')
        if (self.log_state==False) and (self.last_log_state==True): #Stop logging data
            with open(self.event_path, 'a') as file:
                file.write(str(dt.datetime.now())+': Data logging terminated\n')
        if self.endrun_state:
            with open(self.event_path, 'a') as file:
                file.write(str(dt.datetime.now())+': Run ended\n')
                file.close
        self.last_mode_state = self.mode_state
        self.last_log_state = self.log_state
        self.last_endrun_state = self.endrun_state
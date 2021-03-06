# Autonomous_Car
This project consists of a Raspberry Pi controlled RC car that utilizes machine learning to follow lines. A Raspberry Pi 3B was used to control a 1/10th scale Exceed RC truck. The program, written in python, allows for manual and autonomous modes. The autonomous potions features a Tensorflow-0.12.1 model to compute motor controls.
## Setup
Install operation system and all required packages on the Radpberry Pi. Upload the code to the Arduino. Add the Arduino, Raspberry Pi and motor control board between the RC receiver and the motors.
### Hardware
 * 1/10th Scale Exceed RC Truck (must have seperate radio receiver and motor controller)
 * Raspberry Pi 3B
 * Micro SD Card (I recommend 32GB)
 * Arduino Uno
 * Computer with Arduino IDE, Disk Imager and VNC Viewer software
 * Adafruit PCA9685 Motor Control Board
 * Adafruit MMA8451 Accelerometer
 * Raspberry Pi Camera
 * HC-SR04 Sonar Sensor
 * Push Buttons
 * Assorted Wires
 * Battery Pack and micro USB adapter (5V and 2+ Amp output)
 
 Using the Raspberry Pi as a Computer:
 * Monitor with HDMI port
 * USB Bluetooth Combined Keyboard and Mouse
 
 ### Raspbian on Raspberry Pi
 Raspbian is one of the most popular operating systems for the Raspberry Pi. It is a Linux based OS that comes pre-loaded with python, java, mathematica and other programs. Raspbian is available at: https://www.raspberrypi.org/downloads/raspbian/. Follow the instuctions for downloading onto another computer and burning the disk image to the SD card. SD cards 64GB and larger may require re-formatting before installing Raspbian.The Pi used in this project had Raspbian GNU/Linux 8 (jesse) installed. 
 
 ### Raspberry Pi Python Packages
 Attach keyboard, mouse, monitor and connect the Pi to power. Connect to wifi. Enter the command window and run the following command to run python 2:
 ```
 pi@raspberrypi: ~ $ python
 ```
 This will show the version of python 2. The command line prompt will show: '>>>' to indicate the python command line. Python commands can be run from here. Run the exit() command at any time to leave the python session. To run python 3, run the command:
  ```
 pi@raspberrypi: ~ $ python3
 ```
 Python version 3.4 was used in this project. Install all python packages required for this project.
 ```
 pi@raspberrypi: ~ $ sudo apt-get install python3-pip
 ```
 Use the pip command to install the following packages:
 * Adafruit-GPIO (1.0.3)
 * Adafruit-PCA9685 (1.0.1)
 * Adafruit-PureIO (0.2.1)
 * numpy (1.8.2)
 * Pillow (2.6.1)
 * picamera (1.13)
 * pyserial (2.6)
 * RPi.GPIO (0.6.3)
 * matplotlib

To download and install Tensorflow, run the following command:
 ```
 pi@raspberrypi: ~ $ pip3 install tensorflow-0.12.1-cp34-cp34m-linux_armv7l.whl
 ```
 Newer versions of Tensorflow are available. This code has only been tested with Tensorflow-0.12.1.
 
 ### Arduino Libraries
 Download the folowing Arduino libraries:
  * Adafruit MMA8451
  * Adafruit Sensor
  * NewPing (1.9.0)
  
 Add zip libraries from the Arduino IDE. Upload the Arduino code found in this repository to the Arduino Uno.
  
 ### Car and Track
 A track was created by applying white tape to asphalt for high contrast. Data were collected while manually steering the car on the track. The goal was to train the car to stay on the path using tensorflow.
 
 ![alt text](https://github.com/JakeSigwart/Autonomous_Car/blob/master/20180720_115619%5B1%5D.jpg)
 
 ### Results
 Training was tedious, having to be done with small batches and few iterations before saving the model. Training sessions longer than about 30 batches caused the Pi to crash unexpectedly. The results of the training were disappointing as the model anticipated low or zero speed and centered steering for all image inputs. This was the result despite removing all zero-throttle data points.
 
 ### Conclusions
 A more diverse and larger training set may be required to improve results. The learning rate may have to be decreased. More thorough checks have to be in place to remove data points collected when the car is stationary. The acceleration and sonar data are not utalized in the tensorflow graph. The additional data may improve results.

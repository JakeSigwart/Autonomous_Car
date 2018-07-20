import numpy as np

#Input: value (int input(s)),  array (integers that map to: [0-index, mid-index, lest-index]),
#       num_classes (dimension of hot vectors)
#Processing: fix bad values, generate array of zeros
#Output: One-hot array of shape=[num_entries, num_classes]
def one_hot_encode(value, array, num_classes=31, fix_tolerance=20):
    if isinstance(value, int):
        value = np.array([value], dtype=np.float32)
        value = np.reshape(value, (1,1))
        num = 1
    if isinstance(value, list):
        num = len(value)
        value = np.array(value, dtype=np.float32)
        value = np.reshape(value, (num,1))
    if 'numpy' in str(type(value)):
        if value.ndim==1:
            num = value.shape[0]
            value = np.reshape(value, (num,1))
        else:
            value = value.flatten()
            num = value.shape[0]
    
    output = np.zeros([num, num_classes], dtype=np.float32)
    
    ratio = (array[2]-array[0])/(num_classes-1)
    for n in range(0, num):
        val = value[n]
        #Fix bad values
        if val<(array[0]-fix_tolerance) or val>(array[2]+fix_tolerance) or val==0:
            val = array[1]
        index = int(( val - array[0])/ratio + 0.5)
        output[n, index] = 1.00
    return output
    

#Input: value (one-hot array(s)),  array (int values to map to)
#Output: output (array of integers
def one_hot_decode(value, array, num_classes=31):
    ratio = (array[2]-array[0])/(num_classes-1)

    if 'numpy' in str(type(value)):
        dims = value.ndim
        if dims==2:
            num = value.shape[0]
            output = np.zeros([num], dtype=np.int32)
            for n in range(0, num):
                val = value[n]
                index = np.argmax(val, axis=0)                
                output[n] = int(index*ratio + array[0])
            if num==1:
                output = output[0]
        elif dims==1:
            index = np.argmax(value, axis=0)
            output = int( index*ratio + array[0])
            
        else:
            print('Input to : one_hot_decode has wrong number of dimentions.')
            output = []
    else:
        print('Input to: one_hot_decode is unsupported type.')
        output = []
    return output
    
#Input: array of unsigned int images
#Output: array of float images ready for machine learning
def normalize_images(images, span=1.0, min_val=0.0):
    images_out = span*(np.array(images, dtype=np.float32) / 255.0)  + min_val
    if images.ndim==3:
        images_out = np.reshape(images_out, (1, images.shape[0], images.shape[1], images.shape[2]) )
    return images_out



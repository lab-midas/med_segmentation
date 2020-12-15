import numpy as np
#import matplotlib.pyplot as plt
#import tensorflow as tf

#from tensorflow.keras.layers import *
#from tensorflow.keras.optimizers import *
#import tensorflow.keras.backend as K
from med_io.read_HD5F import *

rootdir_file = '/mnt/data/rawdata/Melanom/Tumorvolume'
Data_Reader = HD5F_Reader(dataset='MELANOM', rootdir_file=rootdir_file)

img_IDs = Data_Reader.img_IDs
file = Data_Reader.file
img = file['image']

img_h5 = file['image']['96c12057c3']
mask_h5 = file['mask']['96c12057c3']

print("Shape of the image is: ", img_h5.shape)
print("Shape of the mask_h5 is: ", mask_h5.shape)

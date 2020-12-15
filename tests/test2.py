import numpy as np
#import matplotlib.pyplot as plt
#import tensorflow as tf

#from tensorflow.keras.layers import *
#from tensorflow.keras.optimizers import *
#import tensorflow.keras.backend as K
from med_io.read_HD5F import *


def normalize(img, globalscale=False, channel_at_beginning=False):
    """
    Change pixel values in img to (0,1)
    :param img: type ndarray: input images
    :param globalscale: type boolean: (True) perform normalization on whole 2D/3D image, (False) axis independent normalization
    :param channel_at_beginning: type boolean: channels are at the beginning of input shape
    :return:img: type ndarray
    """

    print("Shape to normalize is: ", img.shape)

    if not channel_at_beginning:

        if globalscale:
            maxval = np.amax(img)
            minval = np.amin(img)
            img = (img - minval) / (maxval - minval + 1E-16)


        else:
            img = [(img[..., i] - np.min(img[..., i])) / (np.ptp(img[..., i]) + 1E-16) for i in
                   range(img.shape[-1])]
        img = np.rollaxis(np.float32(np.array(img)), 0, 4)

    else:  # this means that the channel is on the first position (channels, x, y, z)

        num_channels = img.shape[3]
        # img = np.rollaxis(np.float32(np.array(img)), 0, 4) # changes the channel axis to the end
        print("Shape before normalization: ", img.shape)

        if globalscale:
            maxval = np.amax(img)
            minval = np.amin(img)
            img = (img - minval) / (maxval - minval + 1E-16)


        else:
            img = [(img[..., i] - np.min(img[..., i])) / (np.ptp(img[..., i]) + 1E-16) for i in
                   range(img.shape[-1])]

        img = np.rollaxis(np.float32(np.array(img)), 0, 4)
        print("Shape after normalization: ", img.shape)

        assert num_channels == img.shape[-1], "normalization was not good performed"

    print("Final Shape: ", img.shape)

    return img

def calculate_max_shape(max_shape, img_data):
    img_shape = np.array(img_data.shape)
    if max_shape is None:
        return img_shape
    else:

        assert max_shape.shape == img_shape.shape
        for i in range(len(img_shape)):
            if max_shape[i] < img_shape[i]:
                max_shape[i] = img_shape[i]
    return max_shape

rootdir_file = '/mnt/data/rawdata/Melanom/Tumorvolume'
Data_Reader = HD5F_Reader(dataset='MELANOM', rootdir_file=rootdir_file)

img_IDs = Data_Reader.img_IDs
file = Data_Reader.file
img = file['image']

img_h5 = file['image'][img_IDs[0]]
print("Shape of the image is: ", img_h5.shape)
print("Type of the image is: ", type(img_h5))
# the form of the images are  (channel, H, W, D)

mask_h5 = file['mask_iso'][img_IDs[0]] # mask or label
print("Shape of the mask_h5 is: ", mask_h5.shape)
print("Type of the mask is: ", type(mask_h5))

img_array = np.rollaxis(np.float32(np.array(img_h5)), 0, 4)
print("Shape of the image ARRAY is: ", img_array.shape)
print("Type of the image rolled is: ", type(img_array))

mask_array = np.rollaxis(np.int32(np.array(mask_h5)), 0, 4)
print("Shape of the mask ARRAY is: ", mask_array.shape)
print("Type of the mask rolled is: ", type(mask_array))
n_classes ={}
#print(mask_array)
for i in range(199):
    for j in range(199):
        for k in range(283):
            elem=mask_array[i, j, k,0]

            if str(elem) not in list(n_classes.keys()):
                n_classes[str(elem)] = 1
            else:
                n_classes[str(elem)] = n_classes[str(elem)]+1


print(n_classes)
assert sum(n_classes.values()) == 199*199*283

#max_shape_img, max_shape_label = None, None

#max_shape_img = calculate_max_shape(max_shape_img, img_array)
#print("MaxShape of the image is: ", max_shape_img)
#max_shape_mask = calculate_max_shape(max_shape_label, mask_array)
#print("MaxShape of the mask is: ", max_shape_mask)
#img_normalized = normalize(img_array, channel_at_beginning=True).astype(np.float32)
#print("The normalized shape is: ", img_normalized.shape)

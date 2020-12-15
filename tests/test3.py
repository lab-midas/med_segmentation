import numpy as np
#import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
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

rootdir_file = '/mnt/data/rawdata/Melanom/Tumorvolume'
Data_Reader = HD5F_Reader(dataset='MELANOM', rootdir_file=rootdir_file)

img_IDs = Data_Reader.img_IDs
file = Data_Reader.file
img = file['image']

img_h5 = file['image'][img_IDs[0]]
print("Shape of the image is: ", img_h5.shape)
print("Type of the image is: ", type(img_h5))
# the form of the images are  (channel, H, W, D)

img_array = np.rollaxis(np.float32(np.array(img_h5)), 0, 4)
im_norm = normalize(img_array)
print("Shape of the image ARRAY is: ", im_norm.shape)
print("Type of the image rolled is: ", type(im_norm))
im_norm = normalize(img_array)

mask_h5 = file['mask'][img_IDs[0]] # mask or label
print("Shape of the mask_h5 is: ", mask_h5.shape)
print("Type of the mask is: ", type(mask_h5))

mask_array = np.rollaxis(np.float32(np.array(mask_h5)), 0, 4)
print("Shape of the mask ARRAY is: ", mask_array.shape)
print("Type of the mask rolled is: ", type(mask_array))
n_classes = 2

mask_one_hot = tf.one_hot(tf.argmax(mask_array, axis=-1), 2)
#print("Shape of the mask one hot is: ", mask_one_hot.shape)
#print("Type of the mask one hot is: ", type(mask_one_hot))

#print("first line mask array: ", mask_array[120:130, 120:130, 120:130, 0])
#print("first line one hot tensor: ", mask_one_hot[120:130, 120:130, 120:130, 0])
#print(n_classes)

mask_array_new = mask_array * 0.95

y_true = mask_one_hot
y_pred = im_norm * 0.67
smooth = K.epsilon()
sum_loss, weight_sum = 0, 0
loss_channel_weight = [1.0,1.0]
for class_index in range(2):
    y_t = y_true[..., class_index]
    y_p = y_pred[..., class_index]
    intersection = K.sum(K.abs(y_t * y_p), axis=-1)
    loss = 1 - (2. * intersection + smooth) / (K.sum(K.square(y_t), -1) + K.sum(K.square(y_p), -1) + smooth)
    sum_loss += loss * loss_channel_weight[class_index]
    weight_sum += loss_channel_weight[class_index]

print("sum loss:", sum_loss)
print("dice loss:", sum_loss / (weight_sum + smooth))
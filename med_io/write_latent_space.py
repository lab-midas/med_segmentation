#def write_latent_space(name_layer='bottleneck'):


 #   print("hello world")
  #  print("hello world 2")

import h5py, numpy as np, matplotlib.pyplot as plt
from matplotlib import cm
from keras.utils.np_utils import to_categorical
import tensorflow as tf
#write_latent_space()
print("starting processing")

print("starting processing")
path = '/mnt/data/rawdata/Melanom/Tumorvolume/TUE0000ALLDS_3D.h5'
fileh5 = h5py.File(path, 'r')
file = fileh5['mask']
print("starting processing")
key1 = list(file.keys())[0]
img1_ch = file[key1]
img1 = np.array(img1_ch[:])
#print(img1.shape)
#print(type(img1))
#print("starting processing")
#im = img1[1, :,:,:]
#print(im.shape)
#print(type(im))
#plt.imshow(im[:,:,5])
#plt.colorbar()
#print("starting processing")
#plt.plot()

###----------------------------------------------------------------------------------------------------------
img = np.rollaxis(np.float32(np.array(img1)), 0, 4)
print(img.shape)
print(type(img))
from skimage.transform import resize
IMG_DIM = 50

def normalize(arr):
    arr_min = np.min(arr)
    return (arr-arr_min)/(np.max(arr)-arr_min)

def explode(data):
    shape_arr = np.array(data.shape)
    size = shape_arr[:3] * 2 - 1
    exploded = np.zeros(np.concatenate([size, shape_arr[3:]]), dtype=data.dtype)
    exploded[::2, ::2, ::2] = data
    return exploded


def expand_coordinates(indices):
    x, y, z = indices
    x[1::2, :, :] += 1
    y[:, 1::2, :] += 1
    z[:, :, 1::2] += 1
    return x, y, z

def scale_by(arr, fac):
    mean = np.mean(arr)
    return (arr-mean)*fac + mean


transformed = np.clip(scale_by(np.clip(normalize(img[:, :, : ,0])-0.1, 0, 1)**0.4, 2)-0.1, 0, 1)
resized = resize(transformed, (IMG_DIM, IMG_DIM, IMG_DIM), mode='constant')


def plot_cube(cube, angle=320):
    cube = normalize(cube)

    facecolors = cm.viridis(cube)
    facecolors[:, :, :, -1] = cube
    facecolors = explode(facecolors)

    filled = facecolors[:, :, :, -1] != 0
    x, y, z = expand_coordinates(np.indices(np.array(filled.shape) + 1))

    fig = plt.figure(figsize=(30 / 2.54, 30 / 2.54))
    ax = fig.gca(projection='3d')
    ax.view_init(30, angle)
    ax.set_xlim(right=IMG_DIM * 2)
    ax.set_ylim(top=IMG_DIM * 2)
    ax.set_zlim(top=IMG_DIM * 2)

    ax.voxels(x, y, z, filled, facecolors=facecolors, shade=False)
    plt.show()


plot_cube(resized[:,:,:])

def convert_integers_to_onehot(img, num_classes=3):
    # if some values in img > num_classes-1=> error
    return to_categorical(img, num_classes=num_classes)

def num_classes_one_hot(img):
    count_label = tf.reduce_sum(img, axis=-1)

print("onehot from mask is: ", num_classes_one_hot(img))


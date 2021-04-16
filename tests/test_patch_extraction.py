import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
from med_io.read_HD5F import *
import pickle


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


def read_file():
    rootdir_file = '/mnt/data/rawdata/Melanom/Tumorvolume'
    Data_Reader = HD5F_Reader(dataset='MELANOM', rootdir_file=rootdir_file)

    img_IDs = Data_Reader.img_IDs
    file = Data_Reader.file
    return file, img_IDs


def read_get_image(file, img_IDs, index=0):
    rootdir_file = '/mnt/data/rawdata/Melanom/Tumorvolume'
    Data_Reader = HD5F_Reader(dataset='MELANOM', rootdir_file=rootdir_file)

    img_IDs = Data_Reader.img_IDs
    file = Data_Reader.file
    img = file['image']

    img_h5 = file['image'][img_IDs[index]]
    print("Shape of the image is: ", img_h5.shape)
    print("Type of the image is: ", type(img_h5))
    # the form of the images are  (channel, H, W, D)

    img_array = np.rollaxis(np.float32(np.array(img_h5)), 0, 4)
    im_norm = normalize(img_array)
    print("Shape of the image ARRAY is: ", im_norm.shape)
    print("Type of the image rolled is: ", type(im_norm))
    im_norm = normalize(img_array)

    mask_h5 = file['mask'][img_IDs[index]]  # mask or label
    print("Shape of the mask_h5 is: ", mask_h5.shape)
    print("Type of the mask is: ", type(mask_h5))

    mask_array = np.rollaxis(np.float32(np.array(mask_h5)), 0, 4)
    print("Shape of the mask ARRAY is: ", mask_array.shape)
    print("Type of the mask rolled is: ", type(mask_array))

    # mask_one_hot = tf.one_hot(tf.argmax(mask_array, axis=-1), 2)
    # mask_one_hot = tf.keras.utils.to_categorical(mask_array, num_classes=2)

    # assert (img_array.shape == mask_one_hot.shape)
    # return img_array, mask_one_hot
    return img_array, mask_array


def get_labeled_position(label, class_value, label_any=None):
    """Sample valid idx position inside the specified class.

    Sample a position inside the specified class.
    Using pre-computed np.any(label == class_value, axis=2)
    along third axis makes sampling more efficient. If there
    is no valid position, None is returned.
    Args:
        label (np.array): array with label information H,W,D
        class_value (int): value of specified class
        label_any (list): pre-computed np.any(label == class_value, axis=2)

    Returns:
        list: indices of a random valid position inside the given label
    """
    if label_any is None:
        label_any = np.any(label == class_value, axis=2)

        # Are there any positions with label == class_value?
    valid_idx = np.argwhere(label_any == True)
    if valid_idx.size:
        # choose random valid position (2d)
        rnd = np.random.randint(0, valid_idx.shape[0])
        idx = valid_idx[rnd]
        # Sample additional index along the third axis(=2).
        # Voxel value should be equal to the class value.
        valid_idx_x = label[idx[0], idx[1], :]
        valid_idx_xx = np.argwhere(valid_idx_x == class_value)
        valid_idx_xxx = valid_idx_xx[0]
        rnd = np.random.choice(valid_idx_xxx)
        idx = [idx[0], idx[1], rnd]
    else:
        idx = None

    return idx


def get_random_patch_indices(patch_size, img_shape, pos=None):
    # 3d - image array should have shape H,W,D
    # if idx is given, the patch has to surround this position
    if pos:
        pos = np.array(pos, dtype=np.int)
        min_index = np.maximum(pos - patch_size + 1, 0)
        max_index = np.minimum(img_shape - patch_size + 1, pos + 1)
    else:
        min_index = np.array([0, 0, 0])
        max_index = img_shape - patch_size + 1

    # create valid patch boundaries
    index_ini = np.random.randint(low=min_index, high=max_index)
    index_fin = index_ini + patch_size

    return index_ini, index_fin


def get_slices(img, slice_cut='s', patch=96):
    slices = []
    patches = np.linspace(0, patch - 1, int(patch / 10))
    if slice_cut == 's':
        for index in patches:
            slices.append(img[:, :, int(index)])

    if slice_cut == 'a':
        for index in patches:
            slices.append(img[:, int(index), :])

    if slice_cut == 'c':
        for index in patches:
            slices.append(img[int(index), :, :])

    return slices


def plot_img(img, slice_cut, patch):
    img_slices = get_slices(img, slice_cut, patch=patch)
    for slice in img_slices:
        fig, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].imshow(slice)
        ax[0].set_title(slice_cut + " normal")
        ax[1].imshow(slice, cmap='gray')
        ax[1].set_title(slice_cut + " gray")
        plt.show()


def test_img_patch():
    indexes = [50, 100, 150, 200, 250]
    file, img_IDs = read_file()

    class_probabilities = [0.9, 0.1]
    max_class_value = len(class_probabilities)

    pos_1 = []
    patch_size = np.array([96, 96, 96])
    slice_cut = 'a'

    _label_ax2_any = []

    for id, index in enumerate(indexes):
        img, mask = read_get_image(file, img_IDs, index=index)

        val = np.unique(mask[..., -1])
        _label_ax2_any.append([np.any(mask[..., -1] == c, axis=2) for c in range(max_class_value)])
        val_2 = np.unique(_label_ax2_any[id][0])
        val_3 = np.unique(_label_ax2_any[id][1])
        # selected_class = 1  # np.random.choice(range(len(class_probabilities)), p=class_probabilities)
        selected_class = np.random.choice(range(len(class_probabilities)), p=class_probabilities)
        pos = None
        # get a random point inside the given label
        # if selected_class == 0, use a random position
        if selected_class > 0:
            pos = get_labeled_position(mask[..., -1], selected_class,
                                       label_any=_label_ax2_any[id][selected_class])

        index_ini, index_fin = get_random_patch_indices(patch_size, np.array(mask.shape[:-1]), pos=pos)

        plot_img(mask[index_ini[0]:index_fin[0],
                 index_ini[1]:index_fin[1],
                 index_ini[2]:index_fin[2], -1], slice_cut, patch=patch_size[0])

        patch_img = img[index_ini[0]:index_fin[0],
                    index_ini[1]:index_fin[1],
                    index_ini[2]:index_fin[2], ...]
        patch_mask = mask[index_ini[0]:index_fin[0],
                     index_ini[1]:index_fin[1],
                     index_ini[2]:index_fin[2], ...]

        # path_test = './tests/test_img'
        # if not os.path.exists(path_test): os.makedirs(path_test)
        # with open(path_test + '/' + str(img_IDs[index]) + '_elem.pickle', 'wb') as fp:
        # pickle.dump({'image': patch_img, 'mask': patch_mask}, fp, protocol=pickle.HIGHEST_PROTOCOL)


def test_img_patch_tensor():
    indexes = [50, 100, 150, 200, 250]
    file, img_IDs = read_file()

    class_probabilities = [0.9, 0.1]
    max_class_value = len(class_probabilities)

    pos_1 = []
    patch_size = np.array([96, 96, 96])
    slice_cut = 'a'

    _label_ax2_any = []

    for id, index in enumerate(indexes):
        img, mask = read_get_image(file, img_IDs, index=index)
        mask_tensor = tf.convert_to_tensor(mask)

        _label_ax2_any.append([tf.math.reduce_any(mask_tensor[..., -1] == c, axis=2) for c in range(max_class_value)])
        # val_2 = np.unique(_label_ax2_any[id][0])
        # val_3 = np.unique(_label_ax2_any[id][1])
        # selected_class = 1  # np.random.choice(range(len(class_probabilities)), p=class_probabilities)
        selected_class = np.random.choice(range(len(class_probabilities)), p=class_probabilities)
        pos = None
        # get a random point inside the given label
        # if selected_class == 0, use a random position
        if selected_class > 0:
            pos = get_labeled_position(mask[..., -1], selected_class,
                                       label_any=_label_ax2_any[id][selected_class])

        index_ini, index_fin = get_random_patch_indices(patch_size, np.array(mask.shape[:-1]), pos=pos)

        plot_img(mask_tensor[index_ini[0]:index_fin[0],
                 index_ini[1]:index_fin[1],
                 index_ini[2]:index_fin[2], -1], slice_cut, patch=patch_size[0])

        patch_img = img[index_ini[0]:index_fin[0],
                    index_ini[1]:index_fin[1],
                    index_ini[2]:index_fin[2], ...]
        patch_mask = mask[index_ini[0]:index_fin[0],
                     index_ini[1]:index_fin[1],
                     index_ini[2]:index_fin[2], ...]

##-- in case the test is to be performed, uncomment the next line and run just the test file
#test_img_patch()
# test_img_patch_tensor()


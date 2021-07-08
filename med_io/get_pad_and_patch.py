import numpy as np
import random
import tensorflow as tf
import scipy.stats
import scipy.io as sio
from util import get_thresholds



def pad_img_label(config, max_data_size, images_data, images_shape, labels_data=None, labels_shape=None):
    """
    1.Pad the gap between image and label shape of [x,y,z]
    2.Pad the image(label) to the max data shape in order to get fix patches.
    :param config: type dict: config parameter
    :param images_data: type tf.Tensor: input images data
    :param images_shape: type tf.Tensor: shape input images data
    :param labels_data: type tf.Tensor: input output data. None if not pad label
    :param labels_shape: type tf.Tensor: shape of label data. None if not pad label

    :return:images_data: type tf.Tensor: the padded image data
    :return:labels_data: type tf.Tensor: the padded label data
    """
    dim = len(config['patch_size'])
    if labels_shape is not None and labels_data is not None:
        # resize to Gap between images and labels shape, in order to keep the shape of image and label same in [x,y,z].
        shape_diff_half = [(images_shape[i] - labels_shape[i]) // 2 for i in range(dim)]
        paddings_img = [[tf.minimum(shape_diff_half[i], 0)] * 2 for i in range(dim)]
        paddings_label = [[tf.maximum(shape_diff_half[i], 0)] * 2 for i in range(dim)]

        # Padding isn't executed at channel dimension (last dimension),so pad 0 at last channel
        paddings_img.append([0, 0])
        paddings_label.append([0, 0])

        images_data = tf.pad(tensor=images_data, paddings=paddings_img)
        labels_data = tf.pad(tensor=labels_data, paddings=paddings_label)

    # Since only the fixed patches position (but can be randomly shift)
    # in the pipeline, every image(label) is padded to max shape of this dataset.
    sh = tf.shape(images_data)
    paddings = [[(max_data_size[i] - sh[i]) // 2,
                 (max_data_size[i] - sh[i] + 1) // 2] for i in range(dim)]
    paddings.append([0, 0])  # Padding isn't executed at channel dimension (last dimension).
    images_data = tf.pad(tensor=images_data, paddings=paddings)

    if labels_shape is not None and labels_data is not None:
        labels_data = tf.pad(tensor=labels_data, paddings=paddings)
        return images_data, labels_data
    else:
        return images_data



def get_fixed_patches_index(config, max_fix_img_size, patch_size, overlap_rate=0.5, start=None, end=None, shuffle=False,

                            max_patch_num=None):
    """
    Get fixed patches position list by given image size
    since tf.function of pipeline in Tensorflow 2.0 is not allowed to iterate the values in tf.Tensor,
    it cannot iterate the specific (individual and different) image size of each single image.
    Thus a fix grid of patches is created before creating pipeline.
    Note:Image and label must have the same size!
    :param max_fix_img_size: type list of int: size of unpatched image,
                              the length must be greater than or equal to the length of :param: patch_size
    :param patch_size: type list of int: patch size images
    :param overlap_rate: type float or list of float in [0,1), overlape rate between two patches,
                          the list length must be equal to the length of :param: patch_size
    :param start: type int or list of int: start point of patching
                  the list length must be equal to the length of :param: patch_size
    :param end: type int or list of int: end point of patching
                  the list length must be equal to the length of :param: patch_size
    :param shuffle: type bool: True if shuffle the output list
    :param max_patch_num: type int: max number of patches from a unpatched image. max_patch_num=None if take all patches.
    :return: index_list type list of int list: list of patched position.
    """
    dim = len(patch_size)
    if isinstance(overlap_rate, float): overlap_rate = np.array([overlap_rate] * dim)
    if start is None: start = np.array([0] * dim)
    assert (len(start) == len(overlap_rate) == dim)
    patch_size = [tf.math.minimum(max_fix_img_size[i], patch_size[i]) for i in range(dim)]



    end1 = [max_fix_img_size[i] - patch_size[i] for i in range(dim)]  # stop int list
    

    end1 = [max_fix_img_size[i] - patch_size[i] for i in range(dim)]  # int list

    if end is not None:
        for i in range(dim):
            if end[i] > end1[i]: end[i] = end1[i]
    else:
        end = end1
    if not config['patch_probability_distribution']['use']:
        # Patching with tiling method
        step = patch_size - np.round(overlap_rate * np.array(patch_size))
        for st in step:
            if st <= 0: raise ValueError('step of patches must greater than 0.')

        # Sampling patch index with start, end, step
        slice_ = (*[slice(start[i], end[i] + step[i] - 1, step[i]) for i in range(dim)],)
        index_list = np.array(np.mgrid[slice_].reshape(dim, -1).T, dtype=np.int)

        indices_max_bound = [max_fix_img_size[i] - patch_size[i] for i in range(dim)]
        for j, index in enumerate(index_list):
            # Limiting the patching indices
            index_list[j] = [max(min(index[i], indices_max_bound[i]), 0)
                             for i in range(dim)]
    else:
        # patching with probability method
        index_list = [[0] * dim]
        if not max_patch_num: max_patch_num = 1000  # default max patch number
        N = (max_patch_num, 1)
        if config['patch_probability_distribution']['normal']['use']:
            # Patching sampling with truncated normal distribution
            if config['patch_probability_distribution']['normal']['mu']:
                mu = config['patch_probability_distribution']['normal']['mu']
            else:
                mu = (start + end) // 2  # default mean value

            if config['patch_probability_distribution']['normal']['sigma']:
                sigma = config['patch_probability_distribution']['normal']['sigma']
            else:
                sigma = end - start  # default std value
            print(start, end, mu, sigma)

            # Still some problems here, Tensorflow doesn't support type NPY_INT
            lst = [
                scipy.stats.truncnorm.rvs((start[i] - mu[i]) / sigma, (end[i] - mu[i]) / sigma, loc=mu[i],
                                          scale=sigma[i],
                                          size=N)[:, 0] for i in range(dim)].astype(np.int32)  #
            index_list = np.stack(lst, axis=-1).astype(np.int32)

        if config['patch_probability_distribution']['uniform']:
            # Patching sampling with truncated uniform distribution
            lst = [np.random.uniform(start[i], end[i], size=N)[:, 0] for i in range(dim)]  # [:, 0]
            index_list = np.stack(lst, axis=-1).astype(np.int32)
    shuffle = config['index_shuffle']
    if shuffle: np.random.shuffle(index_list)
    if max_patch_num: index_list = index_list[:max_patch_num]
    
    return index_list


def get_predict_patches_index(data_img, patch_size, overlap_rate=0.5, start=None, output_patch_size=None):
    """
    Get predict patches by given image size.

    :param data_img: type ndarray: unpatched image data with channel,
                      if 3D image, then its shape is [height,width,depth,channel].
    :param patch_size: type list of int: patch size images
    :param overlap_rate: type float or list of float in [0,1), overlape rate between two patches,
                          the list length must be equal to the length of :param: patch_size
    :param start: type int or list of int: start point of patching.
                   The list length must be equal to the lengthj of :param: patch_size
    :param output_patch_size： type list of int: Model output size.
    :return: patch_img_collection:
    :return: index_list: type list of int. Position  of the patch.
    """
    dim = len(patch_size)

    if output_patch_size is not None:
        for j in range(dim): assert patch_size[j] >= output_patch_size[j]

    data_size = np.array(data_img).shape
    if isinstance(overlap_rate, float): overlap_rate = np.array([overlap_rate] * dim)
    if start is None: start = np.array([0] * dim)
    assert (len(patch_size) == len(start) == len(overlap_rate) == dim)
    patch_size = [min(data_size[i], patch_size[i]) for i in range(dim)]
    if output_patch_size is None:
        step = patch_size - np.round(overlap_rate * patch_size)*2
    else:
        step = output_patch_size - np.round(overlap_rate * output_patch_size)
    end = [data_size[i] - patch_size[i] for i in range(dim)]
    for st in step:
        if st <= 0: raise ValueError('step of patches must greater than 0.')

    slice_ = (*[slice(start[i], end[i] + step[i] - 1, step[i]) for i in range(dim)],)
    index_list = np.array(np.mgrid[slice_].reshape(dim, -1).T, dtype=np.int)

    indices_max_bound = [data_size[i] - patch_size[i] for i in range(dim)]

    for j, index in enumerate(index_list):
        index_list[j] = np.float32(np.array([index[i] if (indices_max_bound[i] >= index[i] >= 0)
                                             else max(min(index[i], indices_max_bound[i]), 0)
                                             for i in range(dim)]))
    # indexing using function slice for variable dim # indexing last channel by slice(None, None),equivalent to [:]
    patch_img_collection = [
        np.float32(data_img[(*[slice(index[i], index[i] + patch_size[i]) for i in range(dim)]
                              + [slice(None, None)],)])
        for index in index_list]
    return patch_img_collection, index_list


def unpatch_predict_image(data_patches, indice_list, patch_size, unpatch_data_size=None, set_zero_by_threshold=True,
                          threshold=0.1,
                          output_patch_size=None):
    """
    Unpatch the predict image by list of patch images.
    :param data_patches: type list of ndarray:  Patches data.
    :param indice_list: type list of ndarray:  Patches data position.
                        The length of :param indice_list must be equal to length of :param data_patches.
    :param patch_size: type list of int: size of the patch
    :param unpatch_data_size: type list of ndarray. The size of the unpatch image.
                        The dimension must be equal to length(:param patch_size)+1
                        None if the default data size calculated from the :param indict_list is applied.
    :param set_zero_by_threshold: type bool: True of the values of unpatch image are either 1 or 0.
    :param threshold: type bool: Threshold to set values to 1 if :param discrete =True.
    :param output_patch_size： type list of int: Model output size.
    :return: unpatch_img: type ndarray: unpatched image.
    """
    # Data_patches list
    dim = len(patch_size)
    # add end

    data_patch_size = np.array(data_patches[0]).shape

    assert (len(data_patches) == len(indice_list))

    indice_list = np.int32(indice_list)
    if unpatch_data_size is None:
        max_indice = np.amax(np.array(indice_list), axis=0)
        unpatch_data_size = np.int32([max_indice[i] + patch_size[i] for i in range(dim)] + [data_patch_size[-1]])

    # Initialize predict image (unpatch_img)
    predict_img = np.zeros((*unpatch_data_size,))
    unpatch_weight_map = np.ones((*unpatch_data_size,)) * 1E-16
    # Initialize weight to 1 for each patch size
    if output_patch_size is None:
        weight_patch = np.ones((*patch_size,) + (data_patch_size[-1],))
    else:
        weight_patch = np.ones((*output_patch_size,) + (data_patch_size[-1],))
    print('weight_patch.shape', weight_patch.shape)

    for data_patch, index in zip(data_patches, indice_list):

        # Indexing using function slice for variable dim, Indexing last channel by slice(None, None),equivalent to [:]
        if output_patch_size is None:  # if input image shape==output image shape
            # Overlay all patch value on the predict image
            predict_img[
                (*[slice(index[i], index[i] + patch_size[i]) for i in range(dim)] + [slice(None, None)],)] += data_patch
            # Overlay all weight value on the weight map

            unpatch_weight_map[
                (*[slice(index[i], index[i] + patch_size[i]) for i in range(dim)] + [
                    slice(None, None)],)] += weight_patch

        else:  # else if input image shape>=output image shape
            for j in range(dim): assert patch_size[j] >= output_patch_size[j]
            # Gap between input size image and output size image
            diff = (np.array(patch_size) - np.array(output_patch_size)) // 2
            # Overlay all patch value on the predict image
            predict_img[
                (*[slice(index[i] + diff[i], index[i] + diff[i] + output_patch_size[i]) for i in range(dim)] + [
                    slice(None, None)],)] += data_patch
            # Overlay all weight value on the weight map
            unpatch_weight_map[
                (*[slice(index[i] + diff[i], index[i] + diff[i] + output_patch_size[i]) for i in range(dim)] + [
                    slice(None, None)],)] += weight_patch

    unpatch_img = predict_img / unpatch_weight_map
    #va1 = np.unique(unpatch_img, return_counts=True)
    if set_zero_by_threshold:  unpatch_img[unpatch_img < threshold] = 0
    #va2 = np.unique(unpatch_img, return_counts=True)
    return unpatch_img


def prediction_prob(config, patch_prob_img, indice_list):
    """

    :param config:
    :param patch_prob_img: size(patch num1146, class6)
    :param indice_list:
    :return:
    """

    n_classes = config['body_identification_n_classes']
    #  patch_prob_img size(len of indice_list, n_classes)
    # Initialize
    #
    patch_shape = config['patch_size']  # Body identification patch size [1, X,Y]
    # patch_prob_img[0] is total num of patches.,=
    patch_prob_maps = np.zeros([len(indice_list), patch_shape[1], patch_shape[2], n_classes])
    patch_decision_maps = np.zeros([len(indice_list), patch_shape[1], patch_shape[2], n_classes])
    print('patch_prob_maps,line258', patch_prob_maps.shape)

    for i, pos in enumerate(indice_list):
        # one hot matrix
        patch_decision_maps[i, :, :, np.argmax(patch_prob_img[i, :])] += 1

        for class_ in range(n_classes):
            patch_prob_maps[i, :, :, class_] += patch_prob_img[i, class_]

    # sio.savemat('t.mat',{'d':patch_decision_maps,'p':patch_prob_maps})
    prob_map = unpatch_predict_image(patch_prob_maps, indice_list, patch_shape, set_zero_by_threshold=False)
    decision_map = unpatch_predict_image(patch_decision_maps, indice_list, patch_shape, set_zero_by_threshold=False)

    return prob_map, decision_map





def get_patches_data(data_size, patch_size, data_img, data_label, index_list, random_rate=0.3,
                     slice_channel_img=None, slice_channel_label=None, output_patch_size=None, random_shift_patch=True,
                     squeeze_channel=False, squeeze_dimension=None, images_shape=None):
    """
    Get patches from unpatched image and correspondent label by the list of patch positions.

    :param data_size: type ndarray: data size of :param: data_img and :param data_label
    :param patch_size: type list of int: patch size images
    :param data_img:  type ndarray: unpatched image data with channel,
                       if 3D image, then its shape is [height,width,depth,channel].
    :param data_label: type ndarray: unpatch label data  with channel,
                        if 3D image, then its shape is [height,width,depth,channel].
    :param index_list: type list of list of integers: list position of each patch
    :param slice_channel_img： type list of int:  channel indice chosen for model inputs,
            if :param squeeze_channel is true, the img dimension remains same, else reduce 1.
    :param slice_channel_label： type list of int: channel indice chosen for model outputs
    :param output_patch_size： type list of int: model output size
    :param random_rate: type float,rate of random shift of position from  :param index_list. random_rate=0 if no shift.
    :param random_shift_patch: type bool, True if the patches are randomly shift for data augmentation.
    :param squeeze_channel: type bool, True if select image channel. else all channel will be as input if :param slice_channel_img is False.

    :return: patch_img_collection: type list of ndarray with the shape :param patch_size: list of patches images.
    :return: patch_label_collection type list of ndarray with the shape :param patch_size: list of patches labels.
    :return: index_list: type list of int. Position  of the patch.

    """
    dim = len(patch_size)
    indices_max_bound = [data_size[i] - patch_size[i] for i in range(dim)]

    for j, index in enumerate(index_list):

        # Limiting the patching indices
        index_list[j] = [max(min(index[i], indices_max_bound[i]), 0)
                         for i in range(dim)]

        if random_shift_patch:
            # Shift patches indices for data augmentation
            new_index = [
                index[i] + random.randint(int(-patch_size[i] * random_rate / 2), int(patch_size[i] * random_rate / 2))
                for i in range(dim)]
            index_list[j] = [new_index[i] if (indices_max_bound[i] >= new_index[i] >= 0)
                             else max(min(index[i], indices_max_bound[i]), 0)
                             for i in range(dim)]

    # indexing using function slice for variable dim,indexing last channel by slice(None, None),equivalent to [:]
    # Get patch image data

    patch_img_collection = [data_img[(*[slice(index[i], index[i]+ patch_size[i]) for i in range(dim)]
                                       + [slice(None, None)],)]
                            for index in index_list]

    patch_label_collection = None

    if output_patch_size is not None:
        # If input label shape>=output label shape -> Enlarge label patch
        for j in range(dim): assert patch_size[j] >= output_patch_size[j]
        diff = (np.array(patch_size) - np.array(output_patch_size)) // 2

        # Get label data with size= output_patch_size, keep the centre with same as image patch.
        if data_label is not None:
            patch_label_collection = [
                data_label[(*[slice(index[i] + diff[i], index[i] + diff[i] + output_patch_size[i]) for i in range(dim)]
                             + [slice(None, None)],)]
                for index in index_list]
    else:
        # If input label shape==output label shape
        if data_label is not None:
            patch_label_collection = [
                data_label[(*[slice(index[i], index[i] + patch_size[i]) for i in range(dim)]
                             + [slice(None, None)],)]
                for index in index_list]

    # Select channels for input images and labels by the yaml file
    if slice_channel_img is not None:
        if not squeeze_channel:

            # Select the image channel for patching
            patch_img_collection = [tf.stack([img[..., i] for i in slice_channel_img], axis=-1) for img in
                                    patch_img_collection]
        else:

            # Reduce one dimension (especially for network Body Identification)
            patch_img_collection = [img[..., 0] for img in
                                    patch_img_collection]

        if squeeze_dimension is not None:
            patch_img_collection = [img[..., 0, :] for img in patch_img_collection]

    if slice_channel_label is not None:
        # Select the label channel for patching

        patch_label_collection = [tf.stack([label[..., i] for i in slice_channel_label], axis=-1) for label in
                                  patch_label_collection]

        if squeeze_dimension is not None:
            patch_label_collection = [label[..., 0, :] for label in patch_label_collection]

    return patch_img_collection, patch_label_collection, index_list


def get_sampled_patches(patch_size_tensor, img_data, label_data, class_p=None, max_class_value=None,
                        patches_per_subject=10, data_shape=None, dim_patch=3, channel_img=2, channel_label=2,
                        validation_for_1=0):
    """
        Get sampled patches from unpatched image and correspondent label.

        :param patch_size_tensor: type ndarray: patch size as tensor.
        :param img_data: type ndarray: unpatched images, if 3D image, then its shape is [height,width,depth,channel].
        :param label_data:  type ndarray: unpatched label, if 3D image, then its shape is [height,width,depth,channel].
        :param class_p: type float: normalized class probability.
        :param max_class_value: type integer: number of classes in the label.
        :param patches_per_subject： type integer: number of patches to extract from tensor.
        :param data_shape： type array of integer: shape of input tensor.
        :param dim_patch: type integer: dimension of patch size.
        :param channel_img: type integer: number of channels in image.
        :param channel_label: type integer: number of channels in label.
        :param validation_for_1: type integer: 0 or 1. 0 showing no cancer, 1 otherwise.

        :return: patch_img_return: type list of ndarray from images with size equal to patch size.
        :return: patch_label_return: type list of ndarray from label with size equal to patch size.

    """

    list_patches_img = []
    list_patches_label = []
    _label_ax2_any = []

    _label_ax2_any.append([tf.math.reduce_any(label_data[..., 1] == c, axis=2)
                           for c in range(max_class_value)])

    print("label any: ", _label_ax2_any)
    print("first element:")
    print(_label_ax2_any[0])
    label_data_2_channel = label_data[..., 1]

    for patch in range(patches_per_subject):
        pos = None
        min_index_pos = None
        max_index_pos = None
        selected_class = 0
        # selected_class = np.random.choice(range(len(class_p)), p=class_p)
        selected_class = tf.random.categorical(tf.math.log([tf.convert_to_tensor(class_p)]), num_samples=1)
        print(selected_class)
        #selected_class = tf.reshape(selected_class, shape=())
        print(selected_class)
        # selected_class = 1
        print("selected class: ", selected_class)

        def true_selected_class():
            print("looking for a lesion position ...........................................................")

            def true_fn():
                valid_idx = tf.where(_label_ax2_any[0][1] == True)
                idx = tf.random.shuffle(valid_idx, seed=1)
                idx = tf.gather(idx, indices=0)
                # Sample additional index along the third axis(=2).
                # Voxel value should be equal to the class value.
                valid_idx = label_data_2_channel[idx[0], idx[1], :]
                valid_idx = tf.where(valid_idx == 1)
                rnd = tf.random.shuffle(valid_idx, seed=1)
                rnd = tf.unstack(rnd[0], num=1)
                u = tf.unstack(idx, num=2)
                u.append(rnd[0])
                idx_pos = tf.stack(u)
                idx_pos = tf.cast(idx_pos, dtype=tf.int32)

                # return idx
                min_index = tf.math.maximum(tf.math.add(tf.math.subtract(idx_pos, patch_size_tensor), 1), 0)
                max_index = tf.math.minimum(tf.math.add(tf.math.subtract(data_shape, patch_size_tensor), 1),
                                            tf.math.add(idx_pos, 1))

                return min_index, max_index

            def false_fn():
                min_index = tf.convert_to_tensor([0, 0, 0])
                max_index = tf.math.subtract(data_shape, tf.math.add(patch_size_tensor, 1))
                return min_index, max_index

            val_for_1 = tf.math.greater(validation_for_1, 0)
            min_index_pos_true, max_index_pos_true = tf.cond(val_for_1, true_fn, false_fn)

            return min_index_pos_true, max_index_pos_true

        def false_selected_class():
            min_index_pos_false = tf.convert_to_tensor([0, 0, 0])
            max_index_pos_false = tf.math.subtract(data_shape, tf.math.add(patch_size_tensor, 1))

            return min_index_pos_false, max_index_pos_false

        cond_selected_class = tf.math.greater(selected_class, 0)

        min_index_pos, max_index_pos = tf.cond(cond_selected_class, true_selected_class, false_selected_class)

        ## here we get the position of that patch to slice
        index_ini, index_fin = get_random_patch_indices(patch_size_tensor, min_index=min_index_pos, max_index=max_index_pos)

        ## image shape (x, y, z, channel)
        ## label shape (x, y, z, channel)
        img_patch_per_ch = []
        label_patch_per_ch = []
        for ch in range(channel_img):
            img_patch_ch = img_data[index_ini[0]:index_fin[0], index_ini[1]:index_fin[1], index_ini[2]:index_fin[2], ch]
            img_patch_per_ch.append(img_patch_ch)

        for ch in range(channel_label):
            label_patch_ch = label_data[index_ini[0]:index_fin[0], index_ini[1]:index_fin[1],
                             index_ini[2]:index_fin[2], ch]
            label_patch_per_ch.append(label_patch_ch)

        img_patch = tf.stack(img_patch_per_ch)
        img_patch = tf.transpose(img_patch, perm=[1, 2, 3, 0])
        label_patch = tf.stack(label_patch_per_ch)
        label_patch = tf.transpose(label_patch, perm=[1, 2, 3, 0])

        list_patches_img.append(img_patch)
        list_patches_label.append(label_patch)

    patch_img_return = tf.stack(list_patches_img)

    patch_label_return = tf.stack(list_patches_label)

    return patch_img_return, patch_label_return


def get_random_patch_indices(patch_size_tensor, min_index=None, max_index=None):
    """
            Get random position of the patch to slice.

            :param patch_size_tensor: type ndarray: patch size as tensor.
            :param min_index:  type ndarray: minimum index.
            :param max_index: type ndarray: maximum index.

            :return: index_ini: type ndarray: starting index for patch creation.
            :return: index_fin: type ndarray: final index for patch creation.

    """
    # 3d - image array should have shape H,W,D

    # create valid patch boundaries
    a = tf.math.subtract(max_index, min_index)
    i = tf.random.uniform(shape=(), minval=min_index[0], maxval=max_index[0], dtype=tf.int32)
    ii = tf.random.uniform(shape=(), minval=min_index[1], maxval=max_index[1], dtype=tf.int32)
    iii = tf.random.uniform(shape=(), minval=min_index[2], maxval=max_index[2], dtype=tf.int32)
    index_ini = tf.stack([i, ii, iii])
    # index_ini = np.random.randint(low=min_index[0], high=max_index[0])
    index_fin = tf.math.add(index_ini, patch_size_tensor)

    return index_ini, index_fin


def get_grid_patch_sample_test(img_data, label_data, data_shape=None, patch_size=[96, 96, 96], patch_overlap=0.26,
                               dim_patch=3):
    """
            Get patches for evaluation and prediction tasks.

            :param img_data: type ndarray: image data.
            :param label_data:  type ndarray: label data.
            :param data_shape: type ndarray: shape of image
            :param patch_size: type list: patch size.
            :param patch_overlap:  type float: overlap for adjacent patches.
            :param dim_patch: type integer: dimension of patches.

            :return: patches_img: type list of ndarray: image patches.
            :return: patches_label: type list of ndarray: label patches.

    """

    patch_size_array = np.array(patch_size)
    img_size = data_shape
    patch_overlap_array = np.around(patch_size_array * patch_overlap).astype(np.int64)
    cropped_patch_size = patch_size_array - 2 * patch_overlap_array
    n_patches_dim = tf.cast(tf.math.ceil(tf.math.divide(img_size, cropped_patch_size)), dtype=tf.int32)
    overhead = tf.math.subtract(cropped_patch_size,
                                tf.cast(tf.math.floormod(img_size, cropped_patch_size), dtype=tf.int64))

    padded_img = tf.pad(img_data, [[patch_overlap_array[0], patch_overlap_array[0] + overhead[0]],
                                   [patch_overlap_array[1], patch_overlap_array[1] + overhead[1]],
                                   [patch_overlap_array[2], patch_overlap_array[2] + overhead[2]], [0, 0]])
    padded_label = tf.pad(label_data, [[patch_overlap_array[0], patch_overlap_array[0] + overhead[0]],
                                       [patch_overlap_array[1], patch_overlap_array[1] + overhead[1]],
                                       [patch_overlap_array[2], patch_overlap_array[2] + overhead[2]], [0, 0]])

    list_patch_ksizes = [1]
    [list_patch_ksizes.append(patch_size[i]) for i in range(dim_patch)]
    list_patch_ksizes.append(1)

    list_patch_strides = [1]
    [list_patch_strides.append(int(patch_size[i] - 2 * patch_overlap * patch_size[i])) for i in range(dim_patch)]
    list_patch_strides.append(1)

    padded_img = tf.expand_dims(padded_img, 0)
    padded_label = tf.expand_dims(padded_label, 0)

    patches_img = tf.extract_volume_patches(padded_img, ksizes=list_patch_ksizes,
                                            strides=list_patch_strides, padding='VALID')
    patches_label = tf.extract_volume_patches(padded_label, ksizes=list_patch_ksizes,
                                              strides=list_patch_strides, padding='VALID')

    patches_img = tf.reshape(patches_img, [-1, patch_size[0], patch_size[1], patch_size[2], 2])
    patches_label = tf.reshape(patches_label, [-1, patch_size[0], patch_size[1], patch_size[2], 2])
    # patches_img = tf.squeeze(patches_img)
    # patches_label = tf.squeeze(patches_label)

    return patches_img, patches_label


def get_grid_patch_sample(img_data, label_data, data_shape=None, patch_size=[96, 96, 96], patch_overlap=0.26,
                          dim_patch=3, channel_label=2, list_indices_num=324):

    """Generates grid of overlapping patches.
        All patches are overlapping (2*patch_overlap per axis).
        Cropping the original image by patch_overlap.
        The resulting patches can be re-assembled to the
        original image shape.

        Additional np.pad argument can be passed via **kwargs.
        Args:
            img (np.array): CxHxWxD
            patch_size (list/np.array): patch shape [H,W,D]
            patch_overlap (list/np.array): overlap (per axis) [H,W,D]

        Yields:
            np.array, np.array, int: patch data CxHxWxD,
                                     patch position [H,W,D],
                                     patch number
        """
    dim = dim_patch
    patch_size = np.array(patch_size)
    print(patch_size)
    img_size = data_shape
    print(img_size)
    patch_overlap = np.around(patch_size * patch_overlap).astype(np.int64)
    print(patch_overlap)
    cropped_patch_size = patch_size - 2 * patch_overlap
    print(cropped_patch_size)
    n_patches_dim = tf.cast(tf.math.ceil(tf.math.divide(img_size, cropped_patch_size)), dtype=tf.int32)
    print(n_patches_dim)
    overhead = tf.math.subtract(cropped_patch_size,
                                tf.cast(tf.math.floormod(img_size, cropped_patch_size), dtype=tf.int64))
    print(overhead)
    padded_img = tf.pad(img_data, [[patch_overlap[0], patch_overlap[0] + overhead[0]],
                                   [patch_overlap[1], patch_overlap[1] + overhead[1]],
                                   [patch_overlap[2], patch_overlap[2] + overhead[2]], [0, 0]])
    print(padded_img)
    padded_label = tf.pad(label_data, [[patch_overlap[0], patch_overlap[0] + overhead[0]],
                                       [patch_overlap[1], patch_overlap[1] + overhead[1]],
                                       [patch_overlap[2], patch_overlap[2] + overhead[2]], [0, 0]])
    print(padded_label)

    ## get the range positions
    r1 = tf.range(0, tf.math.multiply(n_patches_dim[0], cropped_patch_size[0]), cropped_patch_size[0])
    print(r1)
    r2 = tf.range(0, tf.math.multiply(n_patches_dim[1], cropped_patch_size[1]), cropped_patch_size[1])
    print(r2)
    r3 = tf.range(0, tf.math.multiply(n_patches_dim[2], cropped_patch_size[2]), cropped_patch_size[2])
    print(r3)
    pos_range = tf.stack([r1, r2, r3])
    print(pos_range)
    pos = tf.transpose(pos_range, perm=[1, 0])
    print(pos)

    n_patches = tf.math.multiply(tf.math.multiply(n_patches_dim[0], n_patches_dim[1]), n_patches_dim[2])
    print(n_patches)

    list_patches_img = []
    list_patches_label = []
    count = tf.convert_to_tensor(np.array([0]))

    for i in range(list_indices_num):
        elem = tf.gather(pos, indices=i)

        im = padded_img[elem[0]:tf.math.add(elem[0], patch_size[0]),
             elem[1]:tf.math.add(elem[1], patch_size[1]),
             elem[2]:tf.math.add(elem[2], patch_size[2]), :]
        label = padded_label[elem[0]:tf.math.add(elem[0], patch_size[0]),
                elem[1]:tf.math.add(elem[1], patch_size[1]),
                elem[2]:tf.math.add(elem[2], patch_size[2]), :]

        list_patches_img.append(im)
        list_patches_label.append(label)

        def true_fn():
            a = tf.add(count, 1)
            return a, i

        # @tf.function
        def false_fn():
            a = tf.add(count, 0)
            ii = 324
            return a, ii

        cond = tf.math.less(count, tf.cast(n_patches, dtype=tf.int64))

        count, i = tf.cond(cond, true_fn, false_fn)
        print(i)

    patch_img_return = tf.stack(list_patches_img)
    patch_label_return = tf.stack(list_patches_label)
    return patch_img_return, patch_label_return


def get_fixed_patches_index_evaluation(config, max_fix_img_size, patch_size, overlap_rate=0.5, start=None, end=None,
                                       shuffle=True,
                                       max_patch_num=None):
    """
    Get fixed patches position list by given image size
    since tf.function of pipeline in Tensorflow 2.0 is not allowed to iterate the values in tf.Tensor,
    it cannot iterate the specific (individual and different) image size of each single image.
    Thus a fix grid of patches is created before creating pipeline.
    Note:Image and label must have the same size!
    :param max_fix_img_size: type list of int: size of unpatched image,
                              the length must be greater than or equal to the length of :param: patch_size
    :param patch_size: type list of int: patch size images
    :param overlap_rate: type float or list of float in [0,1), overlape rate between two patches,
                          the list length must be equal to the length of :param: patch_size
    :param start: type int or list of int: start point of patching
                  the list length must be equal to the length of :param: patch_size
    :param end: type int or list of int: end point of patching
                  the list length must be equal to the length of :param: patch_size
    :param shuffle: type bool: True if shuffle the output list
    :param max_patch_num: type int: max number of patches from a unpatched image. max_patch_num=None if take all patches.
    :return: index_list type list of int list: list of patched position.
    """
    dim = len(patch_size)
    if isinstance(overlap_rate, float): overlap_rate = np.array([overlap_rate] * dim)
    if start is None: start = np.array([0] * dim)
    assert (len(start) == len(overlap_rate) == dim)
    patch_size = [tf.math.minimum(max_fix_img_size[i], patch_size[i]) for i in range(dim)]
    end1 = [max_fix_img_size[i] - patch_size[i] for i in range(dim)]  # stop int list
    if end is not None:
        for i in range(dim):
            if end[i] > end1[i]: end[i] = end1[i]
    else:
        end = end1
    if not config['patch_probability_distribution']['use']:
        # Patching with tiling method
        step = patch_size - np.round(overlap_rate * patch_size) * 2
        for st in step:
            if st <= 0: raise ValueError('step of patches must greater than 0.')

        # Sampling patch index with start, end, step
        slice_ = (*[slice(start[i], end[i] + step[i] - 1, step[i]) for i in range(dim)],)
        index_list = np.array(np.mgrid[slice_].reshape(dim, -1).T, dtype=np.int)

    else:
        # patching with probability method
        index_list = [[0] * dim]
        if not max_patch_num: max_patch_num = 1000  # default max patch number
        N = (max_patch_num, 1)
        if config['patch_probability_distribution']['normal']['use']:
            # Patching sampling with truncated normal distribution
            if config['patch_probability_distribution']['normal']['mu']:
                mu = config['patch_probability_distribution']['normal']['mu']
            else:
                mu = (start + end) // 2  # default mean value

            if config['patch_probability_distribution']['normal']['sigma']:
                sigma = config['patch_probability_distribution']['normal']['sigma']
            else:
                sigma = end - start  # default std value
            print(start, end, mu, sigma)

            # Still some problems here, Tensorflow doesn't support type NPY_INT
            lst = [
                scipy.stats.truncnorm.rvs((start[i] - mu[i]) / sigma, (end[i] - mu[i]) / sigma, loc=mu[i],
                                          scale=sigma[i],
                                          size=N)[:, 0] for i in range(dim)].astype(np.int32)  #
            index_list = np.stack(lst, axis=-1).astype(np.int32)

        if config['patch_probability_distribution']['uniform']:
            # Patching sampling with truncated uniform distribution
            lst = [np.random.uniform(start[i], end[i], size=N)[:, 0] for i in range(dim)]  # [:, 0]
            index_list = np.stack(lst, axis=-1).astype(np.int32)

    if shuffle: np.random.shuffle(index_list)
    if max_patch_num: index_list = index_list[:max_patch_num]
    return index_list

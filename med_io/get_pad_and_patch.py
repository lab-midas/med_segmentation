import numpy as np
import random
import tensorflow as tf
import scipy.stats

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


def get_fixed_patches_index(config,max_fix_img_size, patch_size, overlap_rate=0.5, start=None, end=None, shuffle=False,
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
    end1 = [max_fix_img_size[i] - patch_size[i] for i in range(dim)]  # 停止点列表 int list
    if end is not None:
        for i in range(dim):
            if end[i] > end1[i]: end[i] = end1[i]
    else:
        end = end1
    if not config['patch_probability_distribution']['use']:
        # Patching with tiling method
        step = patch_size - np.round(overlap_rate * patch_size)
        for st in step:
            if st <= 0: raise ValueError('step of patches must greater than 0.')

        # Sampling patch index with start, end, step
        slice_ = (*[slice(start[i], end[i] + step[i] - 1, step[i]) for i in range(dim)],)
        index_list = np.array(np.mgrid[slice_].reshape(dim, -1).T, dtype=np.int)

    else:
        # patching with probability method
        index_list = [[0]*dim]
        if not max_patch_num: max_patch_num = 1000 # default max patch number
        N = (max_patch_num, 1)
        if config['patch_probability_distribution']['normal']['use']:
            # Patching sampling with truncated normal distribution
            if  config['patch_probability_distribution']['normal']['mu']:
                mu = config['patch_probability_distribution']['normal']['mu']
            else:
                mu = (start + end) // 2 # default mean value

            if  config['patch_probability_distribution']['normal']['sigma']:
                sigma = config['patch_probability_distribution']['normal']['sigma']
            else:
                sigma = end - start # default std value
            print(start,end,mu,sigma)

            # Still some problems here, Tensorflow doesn't support type NPY_INT
            lst = [
                scipy.stats.truncnorm.rvs((start[i] - mu[i]) / sigma, (end[i] - mu[i]) / sigma, loc=mu[i],
                                          scale=sigma[i],
                                          size=N)[:, 0] for i in range(dim)].astype(np.int32) #
            index_list = np.stack(lst, axis=-1).astype(np.int32)

        if config['patch_probability_distribution']['uniform']:
            # Patching sampling with truncated uniform distribution
            lst = [np.random.uniform(start[i], end[i], size=N)[:, 0] for i in range(dim)] # [:, 0]
            index_list = np.stack(lst, axis=-1).astype(np.int32)

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
        step = patch_size - np.round(overlap_rate * patch_size)
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


def unpatch_predict_image(data_patches, indice_list, patch_size, unpatch_data_size=None, set_zero_by_threshold=True, threshold=0.1,
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
    if set_zero_by_threshold:  unpatch_img[unpatch_img < threshold] = 0
    return unpatch_img


def get_patches_data(data_size, patch_size, data_img, data_label, index_list, random_rate=0.3,
                     slice_channel_img=None, slice_channel_label=None, output_patch_size=None, random_shift_patch=True,
                     squeeze_channel=False):
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
    patch_img_collection = [
        data_img[(*[slice(index[i], index[i] + patch_size[i]) for i in range(dim)]
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

    if slice_channel_label is not None:
        # Select the label channel for patching
        patch_label_collection = [tf.stack([label[..., i] for i in slice_channel_label], axis=-1) for label in
                                  patch_label_collection]
    return patch_img_collection, patch_label_collection, index_list

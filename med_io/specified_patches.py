import numpy as np

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
        valid_idx = label[idx[0], idx[1], :]
        valid_idx = np.argwhere(valid_idx == class_value)[0]
        rnd = np.random.choice(valid_idx)
        idx = [idx[0], idx[1], rnd]
    else:
        idx = None

    return idx


def get_random_patch_indices(patch_size, img_shape, pos=None):
    """ Create random patch indices.

    Creates (valid) max./min. corner indices of a patch.
    If a specific position is given, the patch must contain
    this index position. If position is None, a random
    patch will be produced.

    Args:
        patch_size (np.array): patch dimensions (H,W,D)
        img_shape  (np.array): shape of the image (H,W,D)
        pos (np.array, optional): specify position (H,W,D), wich should be
                    included in the sampled patch. Defaults to None.

    Returns:
        (np.array, np.array): patch corner indices (e.g. first axis
                              index_ini[0]:index_fin[0])
    """

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
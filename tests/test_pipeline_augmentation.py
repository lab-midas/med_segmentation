import tensorflow as tf
from med_io.pipeline_melanom import *
import matplotlib.pyplot as plt
import pickle
import yaml
import math
from PIL import Image, ImageDraw, ImageFont, ImageShow
import PIL

def get_config(config_path, dataset):
    with open(config_path, "r") as yaml_file:
        config = yaml.load(yaml_file.read(), Loader=yaml.FullLoader)

    filename_max_shape = '.' + config['dir_dataset_info'] + '/max_shape_' + dataset + '.pickle'
    #filename_max_shape = config['dir_dataset_info'] + '/max_shape_' + dataset + '.pickle'

    with open(filename_max_shape, 'rb') as fp:
        config['max_shape'] = pickle.load(fp)

    config['channel_img_num'] = 2
    return config

def get_indexes(path_imgs):
    files = os.listdir(path_imgs)
    indexes = [file.split('_')[1].split('.')[0] for file in files]
    return indexes

def get_paths(config, dataset):
    split_filename = '../' + config['dir_dataset_info'] + '/split_paths_' + dataset + '.pickle'
    #split_filename = config['dir_dataset_info'] + '/split_paths_' + dataset + '.pickle'
    with open(split_filename, 'rb') as fp:
        paths = pickle.load(fp)

    p_keys = paths.keys()
    for key in p_keys:
        print(key)
        for path in paths[key]:
            if "../.." in path[0]:
                break
            path[0] = '../' + path[0]

    return paths

def plot_patch(img, mask, index=0, slice_cut='a', patch=96):
    img_slices = get_slices(img, slice_cut, patch=patch)
    mask_slices = get_slices(mask, slice_cut, patch=patch)


    for slice_img, slice_mask in zip(img_slices, mask_slices):
        fig, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].imshow(slice_img[..., 1])
        ax[0].set_title(slice_cut + " elem: " + str(index) + " image")
        ax[1].imshow(slice_mask[..., 1])
        ax[1].set_title(slice_cut + " elem: " + str(index) + " mask")
        plt.show()

def test_pipeline():
    path_config = '../config/config_melanoma.yaml'
    #path_config = './config/config_melanoma.yaml'
    dataset = 'MELANOM'
    config = get_config(path_config, dataset)
    path_data = get_paths(config, dataset)
    paths_train_img, paths_train_label = path_data['path_train_img'], path_data['path_train_label']
    ds_train = pipeline_melanom(config, paths_train_img, paths_train_label, dataset=dataset, augment=True)
    #with tf.Session() as sess:
    print("starting of trying")
    i=0

    for elem in ds_train:
        print("elem " + str(i) + " ........")
        #print(elem.shape())
        print("image shape: .... ", elem[0].shape)
        print("mask shape: .... ", elem[1].shape)
        values = np.unique(elem[1][..., 1])
        print("mask values: .... ", values)
        #select one random patch in the minibatch and make plots from this
        rnd_batch = np.random.randint(low=0, high=4, size=1)

        #plot_patch((elem[0].numpy())[rnd_batch[0], ...], (elem[1].numpy())[rnd_batch[0], ...],
        #           index=i, slice_cut='a', patch=96)
        #cond = (1.0 in values)
        #if cond:
        plot_combine((elem[0].numpy())[rnd_batch[0], ...], (elem[1].numpy())[rnd_batch[0], ...],
                    index=i, slice_cut='a', patch=96)

        i = i+1
    else:
        print("dataset ran out of data")
    #sess = tf.Session()
    #sess.run(ds_train)
    print("done")

##-- in case the test is to be performed, uncomment the next line and run just the test file
#test_pipeline()
#print("done")
import h5py
import scipy.ndimage.morphology
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.measure import label
from med_io.read_mat import read_mat_file
from med_io.read_HD5F import *
import os
import random
import scipy.io as sio


def lcomp(mask):
    """Computes largest connected component for binary mask.

    Args:
        mask (np.array): input binary mask

    Returns:
        np.array: largest connected component
    """

    labels = label(mask)
    unique, counts = np.unique(labels, return_counts=True)
    # the 0 label is by default background so take the rest
    list_seg = list(zip(unique, counts))[1:]
    largest = max(list_seg, key=lambda x: x[1])[0]
    labels_max = (labels == largest).astype(np.uint8)
    return labels_max

def get_test_keys(config, dataset):
    #path_imgs_predict = config['result_rootdir'] + '/' + config['exp_name'] + '/' + config['model'] + \
    #                    '/predict_result_/' + dataset
    path_imgs_predict = config['result_rootdir'] + '/' + config['exp_name'] + '/' + config['model'] + \
                        '/predict_result_all/' + dataset
    ids_imgs_predict = os.listdir(path_imgs_predict)
    return ids_imgs_predict, path_imgs_predict

def save_postprocessed_image(config, dataset, name_ID, item, data):
    #save_postprocessed_data_dir = config['result_rootdir'] + '/' + config['exp_name'] + '/' + config['model'] + \
    #               '/postprocessing_result/' + dataset + '/' + name_ID
    save_postprocessed_data_dir = config['result_rootdir'] + '/' + config['exp_name'] + '/' + config['model'] + \
                                  '/postprocessing_result_all/' + dataset + '/' + name_ID

    if not os.path.exists(save_postprocessed_data_dir): os.makedirs(save_postprocessed_data_dir)
    save_path = save_postprocessed_data_dir + '/' + 'postprocess_' + config[
        'model'] + '_' + dataset + '_' + name_ID + '.mat'
    sio.savemat(save_path, {item: data})


def plot_mask_compare(mask, mask_post, img, mask_original):
    ## the shape of the mask is (H, W, D, channel)
    assert mask.shape == mask_post.shape
    #the channel is 1

    for w in range(mask.shape[1]):
        fig, ax = plt.subplots(nrows=2, ncols=2)

        ax[0, 0].imshow(img[:, w, :], cmap='gray')
        ax[0, 0].imshow(mask_original[:, w, :], alpha=0.85)
        ax[0, 0].set_title("original mask")
        ax[0,1].imshow(img[:, w, :], cmap='gray')
        ax[0,1].imshow(mask[:, w, :], alpha=0.85)
        ax[0,1].set_title("predicted mask")
        ax[1, 0].imshow(img[:, w, :], cmap='gray')
        ax[1, 0].imshow(mask_post[:, w, :], alpha=0.85)
        ax[1, 0].set_title("postprocess mask")

        plt.show()

def postprocessing(config, datasets):

    for dataset in datasets:
        ids_imgs, path_to_predictions = get_test_keys(config, dataset)
        rootdir_file = config['rootdir_raw_data_img'][dataset]
        Data_Reader = HD5F_Reader(dataset, rootdir_file)

        img_IDs = Data_Reader.img_IDs
        file_keys = Data_Reader.file_keys

        #ids_imgs = ['56ebff0a33']

        #ids_imgs = random.shuffle(ids_imgs)
        i=0
        for id_img in ids_imgs:
            print("image num: ", i)
            print("image id: ", id_img)
            _, predicted_mask, _ = read_mat_file(path=(path_to_predictions + '/' + id_img + '/predict_' + config['model'] +
                                      '_' + dataset + '_' + id_img + '.mat'), read_info=False, read_img=False,
                                labels_name=['predict_image'])

            print("values in predicted mask: ", np.unique(predicted_mask, return_counts=True))
            print("shape of predicted mask: ", predicted_mask.shape)
            print("type of array: ", type(predicted_mask))

            img_h5 = Data_Reader.file[file_keys[0]][id_img]
            img_array = np.rollaxis(np.float32(np.array(img_h5)), 0, 4)
            mask_h5 = Data_Reader.file[file_keys[2]][id_img]
            mask_array_original = np.rollaxis(np.float32(np.array(mask_h5)), 0, 4)

            print("values in original mask: ", np.unique(mask_array_original, return_counts=True))
            print("shape of original mask: ", mask_array_original.shape)
            print("type of array: ", type(mask_array_original))

            if predicted_mask.shape[:-1] != img_array.shape[:-1]:
                ## sometimes after the unpatching, the predicted image is smaller than original
                difference = [(img_array.shape[i] - predicted_mask.shape[i]) for i in range(len(img_array.shape[:-1]))]
                difference.append(1)
                array_difference = [(0, difference[0]), (0, difference[1]), (0, difference[2]), (0,0)]
                new_mask = np.pad(predicted_mask, array_difference, mode='constant', constant_values=0)
                predicted_mask = new_mask
                print("new mask after padding: ", predicted_mask.shape)
                print("values in new predicted mask: ", np.unique(predicted_mask, return_counts=True))
                print("shape of new predicted mask: ", predicted_mask.shape)
                print("type of new array: ", type(predicted_mask))
                ###with this it is ensure that dimensions are good

            threshold = config['threshold']


            # threshold in ct channel of the image
            predicted_mask_ch = np.squeeze(predicted_mask)
            print("values of squeezed predicted mask: ", np.unique(predicted_mask_ch, return_counts=True))
            print("shape of squeezed predicted mask: ", predicted_mask_ch.shape)

            print("values in ct channel img: ", np.unique(img_array[..., 1], return_counts=True))
            print("shape of ct channel img: ", img_array[..., 1].shape)
            print("type of array: ", type(img_array[..., 1]))

            img_th = img_array[..., 1] > threshold
            print("values in threshold img: ", np.unique(img_th, return_counts=True))
            print("shape of threshold img: ", img_th.shape)
            print("type of array: ", type(img_th))

            # fill holes per slice
            for k in range(img_th.shape[2]):
                img_th[:, :, k] = scipy.ndimage.morphology.binary_fill_holes(img_th[:, :, k])
            # keep largest connected component
            img_th = lcomp(img_th)
            print("values in threshold img component: ", np.unique(img_th, return_counts=True))
            print("shape of threshold img component: ", img_th.shape)
            print("type of array: ", type(img_th))

            mask_post = (predicted_mask[..., 0] * img_th).astype(np.uint8)

            print("values in threshold img component: ", np.unique(mask_post, return_counts=True))
            print("shape of threshold img component: ", mask_post.shape)
            print("type of array: ", type(mask_post))
            # plot the mask processed

            #plot_mask_compare(predicted_mask[..., 0], mask_post, img_array[..., 1], mask_array_original[..., 0])

            #save the processed mask into a mat file

            save_postprocessed_image(config, dataset, id_img, 'postprocess_image', mask_post)
            i=i+1
            print("-------------------------------------------------------------------------------")
    print("finished postprocessing")



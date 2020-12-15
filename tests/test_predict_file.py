#from med_io.read_mat import *
import tensorflow as tf
import scipy.io as sio
import os
import numpy as np

def read_mat_file(path=None, channels_img=None,labels_name=None,dim=3,read_label=True, read_img=True,read_info=True):
    """
    Read mat file which contains image data and label data.
    Return images data and labels data with shape(x,y.z, channels)

    :param path: type str, mat file path
    :param imgs_name: type list str, names of channels which are saved in the mat file
    :param labels_name:  type str, img name which are saved in the mat file
    :param
    :return:imgs_data type ndarray, labels_data type ndarray, info: type dict

    """
    info=None
    return_list=[None,None,None]
    if channels_img is None:channels_img = 'predict_image'
    if labels_name is None: labels_name = ['P_BG', 'P_LT', 'P_VAT', 'P_AT']

    patient_dict = sio.loadmat(path)
    img = patient_dict[channels_img]
    # item : info or auxsave must be contained in the mat file.
    #if read_info:
        #info = patient_dict['info'] if 'info' in patient_dict.keys() else patient_dict['auxsave']
    #if read_img:
        # imgs_data image as shape (x, y, z,channels)
        #imgs_data = np.array([patient_dict[channel_img] for channel_img in channels_img])

        # channel at last dimension
        #if len(imgs_data.shape)==dim+1:
            #imgs_data=np.rollaxis(imgs_data, 0, dim+1)
        #  If img_data is 5D, the first dimension is redundant, remove it.
        #if len(imgs_data.shape)==dim+2:imgs_data=imgs_data[0]
        #return_list[0]=np.float32(imgs_data)
    #if read_label:
        # Read label data and set the last dimension to channels
        #labels_data= np.array([patient_dict[label_name] for label_name in labels_name ])
        #labels_data=np.rollaxis(labels_data, 0, dim+1)
        #return_list[1]=np.float32(labels_data)

    #if read_info:
        #info_dict=dict()
        #for info_key in info.dtype.names:
            #info_dict[info_key] =info[0][0][info_key][0]
        #return_list.append(info)

    #return return_list[0],return_list[1],return_list[2]
    return img

def test_predict_file(path, channels_img=None,labels_name=None,dim=3,read_label=True, read_img=True,read_info=True):
    list_mat = read_mat_file(path, channels_img=channels_img,labels_name=labels_name,dim=3,read_label=read_label, read_img=read_img,read_info=read_info)
    return list_mat

def get_classes(list_mat):
    n_classes = {}
    shape = list_mat.shape
    # print(mask_array)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                elem = list_mat[i, j, k]

                if str(elem) not in list(n_classes.keys()):
                    n_classes[str(elem)] = 1
                else:
                    n_classes[str(elem)] = n_classes[str(elem)] + 1

    print(n_classes)
    assert sum(n_classes.values()) == shape[0] * shape[1] * shape[2]
    return n_classes

path1 = '../results_networks/result/Melanoma_1_augmentation/model_U_net_melanoma/predict_result/' \
       'MELANOM/03026f95a3/predict_model_U_net_melanoma_MELANOM_03026f95a3.mat'

list_mat1 = test_predict_file(path1)
get_classes(list_mat1)

path2 = '../results_networks/result/Melanoma_1_augmentation/model_U_net_melanoma/predict_result/' \
       'MELANOM/03426b13fb/predict_model_U_net_melanoma_MELANOM_03426b13fb.mat'

list_mat2 = test_predict_file(path2)
get_classes(list_mat2)

print(list_mat1)
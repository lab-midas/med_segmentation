import pickle
import numpy as np
import os
from .read_mat import *
from .read_dicom import *
from .read_nii import *
from .read_HD5F import *
from .write_tf_record import *
from .read_and_save_datapath import *
from .read_nrrd_path import *
from keras.utils.np_utils import to_categorical

def preprocess_raw_dataset(config):
    """
    Read raw dataset -> parse dataset into tfrecords ->save tfrecords and data info
    Database specific handling
    :param config: type dict, configuring parameters
    :return: None
    """

    global rootdir_tfrec

    def create_tfrec_dir(dir_file, rootdir, rootdir_tfrec, img_folder_name='image', label_folder_name='label',
                         info_folder_name='info'):
        """
        Create tfrecord directories
        :param dir_file: type str: dir of raw data with patient ID
        :param rootdir:  type str: root dir of raw data
        :param rootdir_tfrec: type str, root dir of tfrecords
        :param img_folder_name: type str, name of image subfolder of each dir_file, None if use 'image'
        :param label_folder_name: type str, name of label subfolder of each dir_file, None if use 'label'
        :param info_folder_name: type str, name of info subfolder of each dir_file, None if use 'info'
        :return: dir_tfrec_img: type str, dir_tfrec_label: type str, dir_tfrec_info: type str
        """
        dir_new_file = dir_file.replace(rootdir, rootdir_tfrec).replace('\\', '/')

        dir_tfrec_img = dir_new_file + '/' + img_folder_name
        dir_tfrec_label = dir_new_file + '/' + label_folder_name
        dir_tfrec_info = dir_new_file + '/' + info_folder_name

        if not os.path.exists(dir_tfrec_img): os.makedirs(dir_tfrec_img)
        if not os.path.exists(dir_tfrec_label): os.makedirs(dir_tfrec_label)
        if not os.path.exists(dir_tfrec_info): os.makedirs(dir_tfrec_info)
        return dir_tfrec_img, dir_tfrec_label, dir_tfrec_info

    def write_tfrec_and_pickle_melanoma(imgs_data=None, dir_tfrec_img=None, labels_data=None, dir_tfrec_label=None, info=None,
                               dir_tfrec_info=None, img_tf_name='image', label_tf_name='label', info_name='info'):
        """
        Write images and label pairs into tfrecord files, and save infomation of patient into pickle files
        for Melanoma dataset
        :param imgs_data: type ndarray
        :param dir_tfrec_img:  type str
        :param labels_data:  type ndarray
        :param dir_tfrec_label: type str
        :param info: type dict
        :param dir_tfrec_info: type str
        :return:
        """
        if imgs_data is not None:
            write_tfrecord_melanoma(imgs_data, info['validation_for_cancer'], path=dir_tfrec_img + '/'+img_tf_name+'.tfrecords')
            print("Image succesfully written")

        if labels_data is not None:
            write_tfrecord_melanoma(labels_data, info['validation_for_cancer'], path=dir_tfrec_label + '/'+label_tf_name+'.tfrecords')
            print("Mask succesfully written")

        if info is not None:
            pickle.dump(info, open(dir_tfrec_info + '/'+info_name+'.pickle', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    def write_tfrec_and_pickle(imgs_data=None, dir_tfrec_img=None, labels_data=None, dir_tfrec_label=None, info=None,
                               dir_tfrec_info=None, img_tf_name='image', label_tf_name='label', info_name='info'):
        """
        Write images and label pairs into tfrecord files, and save infomation of patient into pickle files
        :param imgs_data: type ndarray
        :param dir_tfrec_img:  type str
        :param labels_data:  type ndarray
        :param dir_tfrec_label: type str
        :param info: type dict
        :param dir_tfrec_info: type str
        :return:
        """
        if imgs_data is not None:
            write_tfrecord(imgs_data, path=dir_tfrec_img + '/'+img_tf_name+'.tfrecords')
        if labels_data is not None:
            write_tfrecord(labels_data, path=dir_tfrec_label + '/'+label_tf_name+'.tfrecords')
        if info is not None:
            pickle.dump(info, open(dir_tfrec_info + '/'+info_name+'.pickle', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    def normalize(img, globalscale=False):
        """
        Change pixel values in img to (0,1)
        :param img: type ndarray: input images
        :param globalscale: type boolean: (True) perform normalization on whole 2D/3D image, (False) axis independent normalization
        :param channel_at_beginning: type boolean: channels are at the beginning of input shape
        :return:img: type ndarray
        """
        num_channels = img.shape[-1]
        print("Shape to normalize is: ", img.shape)

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

    def save_max_shape(dataset, max_shape_img, max_shape_label=None, save_filename='max_shape'):

        dictionary = {"image": max_shape_img, "label": max_shape_label, "dataset": dataset}
        if not os.path.exists(config['dir_dataset_info']): os.makedirs(config['dir_dataset_info'])

        if config['read_body_identification']:
            pickle_filename = config['dir_dataset_info'] + '/max_shape_' + dataset + '_bi.pickle'
        else:
            pickle_filename = config['dir_dataset_info'] + '/max_shape_' + dataset + '.pickle'

        pickle.dump(dictionary, open(pickle_filename, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    # if item in config are str, change them to list
    if isinstance(config['dataset'], str): config['dataset'] = [config['dataset']]
    if isinstance(config['rootdir_raw_data_img'], str): config['rootdir_raw_data_img'] = [
        config['rootdir_raw_data_img']]
    if isinstance(config['rootdir_raw_data_label'], str): config['rootdir_raw_data_img'] = [
        config['rootdir_raw_data_label']]
    if isinstance(config['rootdir_tfrec'], str): config['rootdir_tfrec'] = [config['rootdir_tfrec']]

    # Keep the images shape constant with labels.
    # image_reshape = lambda x: np.rollaxis(np.rollaxis(np.rollaxis(x, 0, 4), 0, 3), 0, 2)  # shape order(x,y,z,channel)
    image_reshape = lambda x: np.transpose(x, (3, 2, 1, 0))  # shape order(x,y,z,channel)

    for i_dtype, dataset in enumerate(config['dataset']):
        max_shape_img, max_shape_label = None, None

        print('Start processing dataset: ', dataset, ' ...')

        # Adipose Tissue databases
        print('dataset:',dataset)
        if dataset in ['TULIP1_5T', 'NAKO_AT', 'TULIP3T']:
            # load mat file, which the labels and the images data are stored together
            rootdir = config['rootdir_raw_data_img'][
                dataset]  # rootdir info is stored in config['rootdir_raw_data_img']
            rootdir_tfrec = config['rootdir_tfrec'][dataset]

            name_IDs = os.listdir(rootdir)
            for name_ID in name_IDs:
                filenames = os.listdir(rootdir + '/' + name_ID)
                if not filenames:
                    print(dataset, ': ', name_ID, ' has no mat files in ', filenames,
                          'this dataset is abandoned.')
                    break
                paths_mat = [rootdir + '/' + name_ID + '/' + filename for filename in filenames]  # path of mat files
                for path_mat in paths_mat:

                    if not config['read_body_identification']:

                        try:
                            imgs_data, labels_data, info = read_mat_file(path=path_mat)

                        except Exception as e:
                            print(dataset, ': ', 'Loading mat files of ', name_ID,
                                  'failed. This dataset is abandoned. Error info:')
                            print(e)
                            break


                        max_shape_img = calculate_max_shape(max_shape_img, imgs_data)
                        max_shape_label = calculate_max_shape(max_shape_label, labels_data)
                        imgs_data = normalize(imgs_data).astype(np.float32)

                        print(dataset, ': ', name_ID, ' images shape:', imgs_data.shape, ' labels shape:',
                              labels_data.shape)



                        infos = {'name_ID': name_ID,
                                 'info_patient': info,
                                 'name_input_channel': config['name_input_channel'][dataset],
                                 'name_output_channel': config['name_output_channel'][dataset]}

                        # Create tfrecord dirs and save the datasets into tfrecords files.
                        dir_tfrec_img, dir_tfrec_label, dir_tfrec_info = create_tfrec_dir(rootdir + '/' + name_ID,
                                                                                          rootdir,
                                                                                          rootdir_tfrec)
                        write_tfrec_and_pickle(imgs_data, dir_tfrec_img, labels_data, dir_tfrec_label, infos,
                                               dir_tfrec_info)

                        save_max_shape(dataset, max_shape_img, max_shape_label)
                    else:
                        # read body identification data

                        try:
                            imgs_data, labels_data, info = read_mat_file_body_identification(path=path_mat,
                                                                                             read_img=True)
                        except Exception as e:
                            print(dataset, ': ', 'Loading mat files of ', name_ID,
                                  'failed. This dataset is abandoned. Error info:')
                            print(e)
                            break
                        max_shape_img = calculate_max_shape(max_shape_img, imgs_data)
                        max_shape_label = np.array(labels_data.shape)
                        imgs_data = np.float32(normalize(imgs_data))
                        print(dataset, ': ', name_ID, ' images shape:', imgs_data.shape, ' labels shape:',
                              labels_data.shape)

                        infos = {'name_ID': name_ID,
                                 'info_patient': info,
                                 'name_input_channel': config['name_input_channel'][dataset],
                                 'name_output_channel': config['name_output_channel'][dataset]}

                        # Create tfrecord dirs and save the datasets into tfrecords files.
                        dir_tfrec_img, dir_tfrec_label, dir_tfrec_info = create_tfrec_dir(rootdir + '/' + name_ID,
                                                                                          rootdir,
                                                                                          rootdir_tfrec,
                                                                                          label_folder_name='label_bi')
                        write_tfrec_and_pickle(imgs_data, dir_tfrec_img, labels_data, dir_tfrec_label, infos,
                                               dir_tfrec_info)
                        save_max_shape(dataset, max_shape_img, max_shape_label)


        # KORA
        elif dataset == 'KORA':
            rootdir_img = config['rootdir_raw_data_img'][dataset]
            rootdir_label = config['rootdir_raw_data_label'][dataset]
            rootdir_tfrec = config['rootdir_tfrec'][dataset]

            dir_patterns = {'images': '/*/study', 'labels': '/*/*_ROI'}
            name_IDs = sorted(list(set(os.listdir(rootdir_img)).intersection(set(os.listdir(rootdir_label)))))

            channels_image = ['_iso_' + item + '_' for item in config['name_input_channel'][dataset]]
            channels_label = config['name_output_channel'][dataset]

            for name_ID in name_IDs:
                # Read image files
                dir_img = rootdir_img + dir_patterns['images'].replace('*', name_ID)
                channels_img = sorted(os.listdir(dir_img))

                # Since not all the images in channel match the label, only selected image channels are used!
                inds_img = [[i for i in range(len(channels_img)) if m in channels_img[i]] for m in channels_image]
                choose_inds_img = [inds_img[i][0] for i in range(len(inds_img))]
                dir_chosen_channel_img = [dir_img + '/' + channels_img[i] for i in choose_inds_img]

                imgs_data, info = [], []
                for dir_channel in dir_chosen_channel_img:
                    if not os.listdir(dir_channel):
                        print(dataset, ': ', name_ID, ' has no dicom files in ', dir_channel,
                              'this dataset is abandoned.')
                        break

                    try:
                        # Read dicom path and use 'format' to delete *-*-*-0002.dcm file in KORA dataset!
                        img_data, info_patient = read_dicom_dir(dim_dir=dir_channel, format=r"\w{2}-\d{4}-\d{4}.dcm")
                    except Exception as e:
                        print(dataset, ': ', 'Loading dicom files of ', name_ID,
                              'failed. This dataset is abandoned. Error info:')
                        print(e)
                        break
                    imgs_data.append(img_data)
                    info.append(info_patient)
                else:  # if not break by the for loop above
                    imgs_data = image_reshape(normalize(np.array(imgs_data))).astype(np.float32)

                    # Read label files
                    dir_label = rootdir_label + dir_patterns['labels'].replace('*', name_ID)
                    if not os.path.exists(dir_label):
                        # KORA2454788  the label name is different.
                        dir_label = dir_label.replace('_ROI', '_ROi')
                        if not os.path.exists(dir_label):
                            print(dataset, ': ', name_ID, ' has no label dirs. This dataset is abandoned.')
                            continue

                    if os.listdir(dir_label) == []:
                        print(dataset, ': ', name_ID, ' has no label files in ', dir_label,
                              'this dataset is abandoned.')
                        continue
                    try:
                        labels_data = [read_nii_path(path=dir_label + '/' + channel + '.nii') for channel in
                                       channels_label]
                    except Exception as e:
                        print(dataset, ': ', 'Loading label files of ', name_ID,
                              'failed. This dataset is abandoned. Error info:')
                        print(e)
                        continue

                    labels_data = np.rollaxis(np.float32(np.array(labels_data)), 0, 4)  # shape order:[x,y,z,channel]
                    print(name_ID, ': image shape:', imgs_data.shape, ' labels shape:', labels_data.shape)

                    # Sometimes the shape of the labels with 5D matches not the images with 4D,
                    # the dataset of this ID would be abandoned!
                    if imgs_data.shape[0:-1] != labels_data.shape[0:-1] or len(imgs_data.shape) != len(
                            labels_data.shape):
                        print(dataset, ': ', 'shape of ID ', name_ID, 'image: ', imgs_data.shape, ', labels: ',
                              labels_data.shape,
                              'does not match!')
                        continue

                    max_shape_img = calculate_max_shape(max_shape_img, imgs_data)
                    max_shape_label = calculate_max_shape(max_shape_label, labels_data)
                    infos = {'name_ID': name_ID,
                             'info_patient': info,
                             'name_input_channel': config['name_input_channel'][dataset],
                             'name_output_channel': config['name_output_channel'][dataset]}

                    # Create tfrecord dirs and save the datasets into tfrecords files.
                    dir_tfrec_img, dir_tfrec_label, dir_tfrec_info = create_tfrec_dir(
                        dir_file=rootdir_img + '/' + name_ID,
                        rootdir=rootdir_img,
                        rootdir_tfrec=rootdir_tfrec)

                    write_tfrec_and_pickle(imgs_data, dir_tfrec_img, labels_data, dir_tfrec_label, infos,
                                           dir_tfrec_info)
                    save_max_shape(dataset, max_shape_img, max_shape_label)



        # mMR Attenuation masks
        elif dataset == 'DIXON':
            rootdir_img = config['rootdir_raw_data_img'][dataset]
            rootdir_label = config['rootdir_raw_data_label'][dataset]
            rootdir_tfrec = config['rootdir_tfrec'][dataset]

            dir_patterns = {'images': '/*', 'labels': '/*/label'}
            name_IDs = sorted(list(
                set(os.listdir(rootdir_img)).intersection(set(os.listdir(rootdir_label)))))  # name ID of DIXON Dataset
            channels_image = config['name_input_channel'][dataset]
            channels_label = config['name_output_channel'][dataset]
            for name_ID in name_IDs:


                # Read image files
                dir_img = rootdir_img + dir_patterns['images'].replace('*', name_ID)
                dir_chosen_channel_img = [dir_img + '/' + channel_img for channel_img in channels_image]
                imgs_data, info = [], []

                for dir_channel in dir_chosen_channel_img:
                    if not os.listdir(dir_channel):
                        print(dataset, ': ', name_ID, ' has no dicom files in ', dir_channel,
                              'this dataset is abandoned.')
                        break

                    try:
                        img_data, info_patient = read_dicom_dir(dim_dir=dir_channel, all_files=True,order='filename_last_8_character')
                    except Exception as e:
                        print(dataset, ': ', 'Loading dicom files of ', name_ID,
                              'failed. This dataset is abandoned. Error info:')
                        print(e)
                        break
                    imgs_data.append(img_data)
                    info.append(info_patient)
                else:
                    imgs_data = image_reshape(normalize(np.array(imgs_data))).astype(np.float32)
                    # Read label files
                    dir_label = rootdir_label + dir_patterns['labels'].replace('*', name_ID)
                    try:
                        labels_data = [read_nii_path(path=dir_label + '/' + channel + '.nii') for channel in
                                       channels_label]
                    except Exception as e:
                        print(dataset, ': ', 'Loading label files of ', name_ID,
                              'failed. This dataset is abandoned. Error info:')
                        print(e)
                        continue
                    labels_data = np.rollaxis(np.float32(np.array(labels_data)), 0, 4)  # shape order:[x,y,z,channel]

                    print(name_ID, ': image shape:', imgs_data.shape, ' labels shape:', labels_data.shape)
                    # Sometimes the shape of the labels  matches not the images,
                    # the dataset of this ID would be abandoned!
                    if imgs_data.shape[0:-1] != labels_data.shape[0:-1] or len(imgs_data.shape) != len(
                            labels_data.shape):
                        print(dataset, ': ', 'shape of ID ', name_ID, 'image: ', imgs_data.shape, ', labels: ',
                              labels_data.shape,
                              'does not match!')
                        continue
                    max_shape_img = calculate_max_shape(max_shape_img, imgs_data)
                    max_shape_label = calculate_max_shape(max_shape_label, labels_data)
                    infos = {'name_ID': name_ID,
                             'info_patient': info,
                             'name_input_channel': config['name_input_channel'][dataset],
                             'name_output_channel': config['name_output_channel'][dataset]}

                    # Create tfrecord dirs and save the datasets into tfrecords files.
                    dir_tfrec_img, dir_tfrec_label, dir_tfrec_info = create_tfrec_dir(
                        dir_file=rootdir_img + '/' + name_ID,
                        rootdir=rootdir_img,
                        rootdir_tfrec=rootdir_tfrec)

                    write_tfrec_and_pickle(imgs_data, dir_tfrec_img, labels_data, dir_tfrec_label, infos,
                                           dir_tfrec_info)
                    save_max_shape(dataset, max_shape_img, max_shape_label)
                    del imgs_data
                    del labels_data
                    del infos

        # NAKO
        elif dataset == 'NAKO':

            rootdir_img = config['rootdir_raw_data_img'][dataset]
            rootdir_label = config['rootdir_raw_data_label'][dataset]
            rootdir_tfrec = config['rootdir_tfrec'][dataset]

            name_IDs = sorted(list(set([name_ID[:-3] for name_ID in os.listdir(rootdir_img)]).
                                   intersection(set(os.listdir(rootdir_label)))))

            channels_img = ['3D_GRE_TRA_W_COMPOSED']
            channels_label = config['name_output_channel'][dataset]
            dir_patterns = {'images': '', 'labels': '/labels/*_#'}

            for name_ID in name_IDs:
                dir_with_ID = rootdir_img + '/' + str(name_ID) + '_30/' + dir_patterns['images'].replace('*', name_ID)
                imgs_data, info = [], []
                for chn_img in channels_img:
                    #  Only choose '3D_GRE_TRA_W_COMPOSED'.
                    channel_chosen = [chn for chn in os.listdir(dir_with_ID) if chn_img in chn][0]
                    dir_files = dir_with_ID + '/' + channel_chosen
                    if not os.listdir(dir_files):
                        print(dataset, ': ', name_ID, ' has no dicom files in ', chn_img,
                              'this dataset is abandoned.')
                        break
                    try:
                        img_data, info_patient = read_dicom_dir(dim_dir=dir_files, all_files=True, order='NAKO')
                    except Exception as e:
                        print(dataset, ': ', 'Loading dicom files of ', name_ID,
                              'failed. This dataset is abandoned. Error info:')
                        print(e)
                        break
                    imgs_data.append(img_data)
                    info.append(info_patient)
                else:
                    imgs_data = np.array(imgs_data)
                    # Because of speciality of NAKO dataset,
                    # the shape of output from the function 'read_dicom_dir' may be 5 dimensions.
                    # Reduce it to 4 dimensions.
                    if len(imgs_data.shape) == 5: imgs_data = imgs_data[0]
                    imgs_data = image_reshape(normalize(imgs_data)).astype(np.float32)

                    paths_label_with_chns = [rootdir_label + '/' + str(name_ID) + dir_patterns['labels'].
                        replace('*', name_ID).replace('#', channel_label) + '.nrrd' for channel_label in channels_label]

                    try:
                        labels_data = [read_nrrd_path(path_chn)[..., 0] for path_chn in paths_label_with_chns]
                    except Exception as e:
                        print(dataset, ': ', 'Loading labels files of ', name_ID,
                              'failed. This dataset is abandoned. Error info:')
                        print(e)
                        continue

                    labels_data = np.rollaxis(np.float32(np.array(labels_data)), 0, 4)  # shape order:[x,y,z,channel]

                    max_shape_img = calculate_max_shape(max_shape_img, imgs_data)
                    max_shape_label = calculate_max_shape(max_shape_label, labels_data)

                    infos = {'name_ID': name_ID,
                             'info_patient': info,
                             'name_input_channel': config['name_input_channel'][dataset],
                             'name_output_channel': config['name_output_channel'][dataset]}
                    print(name_ID, ': image shape:', imgs_data.shape, ' labels shape:', labels_data.shape)

                    # Create tfrecord dirs and save the datasets into tfrecords files.
                    dir_tfrec_img, dir_tfrec_label, dir_tfrec_info = create_tfrec_dir(
                        dir_file=rootdir_img + '/' + name_ID,
                        rootdir=rootdir_img,
                        rootdir_tfrec=rootdir_tfrec)
                    write_tfrec_and_pickle(imgs_data, dir_tfrec_img, labels_data, dir_tfrec_label, infos,
                                           dir_tfrec_info)
                    save_max_shape(dataset, max_shape_img, max_shape_label)

        ## for Melanom dataset
        elif dataset == 'MELANOM':

            # directories in server for the HD5F files
            # directory of images and masks are the same
            rootdir_file = config['rootdir_raw_data_img'][dataset]
            print("root dir from img and label is: ", rootdir_file)
            #rootdir_label = config['rootdir_raw_data_label'][dataset]
            #print(root_dir_label)
            rootdir_tfrec = config['rootdir_tfrec'][dataset]
            print("root dir for tfrecords is: ", rootdir_tfrec)

            #dir_patterns = {'images': '/*/image', 'mask': '/*/mask'}

            #we get the img_IDs from images, the HD5F file, the keys from file
            #img_IDs, file, file_keys = read_HD5F(dataset, rootdir_file, padded=False)
            Data_Reader = HD5F_Reader(dataset, rootdir_file)

            img_IDs = Data_Reader.img_IDs
            file_keys = Data_Reader.file_keys

            i=0
            ## iterate over the IDs in order to save each image as tfrecord
            for img_ID in img_IDs:


                #for channel in range(num_channels): # (PET, CT)
                print("------------------------------------------------------------------")
                print("ID: ", img_ID)

                img_h5 = Data_Reader.file[file_keys[0]][img_ID]
                print("Shape of the image is: ", img_h5.shape)
                # the form of the images are  (channel, H, W, D)

                mask_type = config['keys_reader'][1]
                if mask_type == 'mask_iso':
                    mask_h5 = Data_Reader.file[file_keys[2]][img_ID] # mask iso
                    print("Shape of the mask_h5 is: ", mask_h5.shape)

                else:
                    mask_h5 = Data_Reader.file[file_keys[1]][img_ID]  # mask
                    print("Shape of the mask_h5 is: ", mask_h5.shape)

                img_array = np.rollaxis(np.float32(np.array(img_h5)), 0, 4)
                print("Shape of the image ARRAY is: ", img_array.shape)

                mask_array = np.rollaxis(np.float32(np.array(mask_h5)), 0, 4)
                print("Shape of the mask ARRAY is: ", mask_array.shape)
                un = np.unique(mask_array)

                ## shape = (x, y, z, 1)

                ##_-----transform the mask into one hot encoded -----------------
                values_b = np.unique(mask_array)
                ##assert values_b == [0.0,1.0], "Labels do not contain only 0s and 1s"
                mask_one_hot = to_categorical(mask_array, num_classes=config['num_classes'])
                #values_1 = np.unique(mask_one_hot[...,0])
                #values_2 = np.unique(mask_one_hot[..., 1])
                assert mask_one_hot.shape[-1] == config['num_classes'], "Mask is not in one hot encoded"

                ##--------------------------------------------------------------
                # some images do not present lesion in the image
                # for further pipeline development we need to validate them
                # if they have lesion or not
                # lesion appears in the mask as 1
                # Non-lesion appears in the mask as 0
                # in case an image has lesion and non-lesion parameters
                # then it must have 2 elements, otherwise just 1

                values_in_array = np.unique(mask_array)
                print("values in mask: ", values_in_array)
                validation_for_cancer = False

                if values_in_array.size == 2:
                    validation_for_cancer = True

                print("validation for cancer: ", validation_for_cancer)
                ##-------------------------------------------------------------------

                max_shape_img = calculate_max_shape(max_shape_img, img_array)
                print("MaxShape of the image is: ", max_shape_img)
                #max_shape_label = calculate_max_shape(max_shape_label, mask_array)
                #print("MaxShape of the mask is: ", max_shape_label)
                max_shape_label = calculate_max_shape(max_shape_label, mask_one_hot)
                print("MaxShape of the mask one hot encoded is: ", max_shape_label)
                #max_shape_mask_iso = calculate_max_shape(max_shape_label, img['mask_iso'][img_ID])
                img_normalized = normalize(img_array).astype(np.float32)
                print("The normalized shape is: ", img_normalized.shape)

                #assert img_normalized.shape[:3] == mask_array.shape[:3], "Dimensions from image and mask do not match"
                assert img_normalized.shape[:3] == mask_one_hot.shape[:3], "Dimensions from image and mask do not match"

                infos = {'name_ID': img_ID,
                         'info_patient': "info",
                         'name_input_channel': config['name_input_channel'][dataset],
                         'name_output_channel': config['name_output_channel'][dataset],
                         'validation_for_cancer': validation_for_cancer}
                #print(img_ID, ': image shape:', img_normalized.shape, ' labels shape:', mask_array.shape)
                print(img_ID, ': image shape:', img_normalized.shape, ' labels shape:', mask_one_hot.shape)

                ## create tfrecord directory
                dir_tfrec_img, dir_tfrec_label, dir_tfrec_info = create_tfrec_dir(
                    dir_file=rootdir_file + '/' + img_ID,
                    rootdir=rootdir_file,
                    rootdir_tfrec=rootdir_tfrec)

                ## write tfrec in pickle file
                write_tfrec_and_pickle_melanoma(imgs_data=img_normalized, dir_tfrec_img=dir_tfrec_img,
                                       labels_data=mask_one_hot, dir_tfrec_label=dir_tfrec_label,
                                       info=infos, dir_tfrec_info=dir_tfrec_info)
                ##save the max shape
                save_max_shape(dataset, max_shape_img, max_shape_label)

                i = i+1

            print("Melanom Dataset preprocessed")

        # read all paths of tfrecords and save into the pickle files
        print(rootdir_tfrec)
        read_and_save_tfrec_path(config, rootdir_tfrec,
                                 filename_tfrec_pickle=config['filename_tfrec_pickle'][dataset],
                                 dataset=dataset)

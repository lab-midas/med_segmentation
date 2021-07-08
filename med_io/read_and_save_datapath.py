import os
import pickle


def read_and_save_tfrec_path(config, rootdir, filename_tfrec_pickle=None, dataset='0'):
    """
    Read all paths of tfrecords and save into the pickle files
    :param rootdir: type str: rootdir of saving tfrecords dataset
    :param filename_tfrec_pickle: type str: Filename of pickle which stores the paths of all tfrecords files.

    :return:
    """
    dirs = os.listdir(rootdir)
    #print(dirs)
    if dirs==[]:
        print('Failed saving tfrecords files: No directories in the tfrecords rootdir!')
        return None
    lst_image = []
    lst_label = []
    lst_info = []

    if config['read_body_identification']:
        dir_patterns = {'images': '/*/image', 'labels': '/*/label_bi', 'info_patient': '/*/info'}
        filename_tfrec_pickle=filename_tfrec_pickle+'_bi'
    else:
        dir_patterns = {'images': '/*/image', 'labels': '/*/label', 'info': '/*/info'}
    for d in dirs:
        dir_label = rootdir + dir_patterns['labels'].replace('*', d)
        dir_image = rootdir + dir_patterns['images'].replace('*', d)
        dir_info = rootdir + dir_patterns['info'].replace('*', d)
        #if not os.path.exists(dir_label) or not os.path.exists(dir_image):
            #print(d, ' is not found in label dir or in image dir of tfrecords,this dataset is abandoned.')
            #continue

        if not os.path.exists(dir_label):
            print(d, ' is not found in label dir of tfrecords,this dataset is abandoned.')
            continue

        if not os.path.exists(dir_image):
            print(d, ' is not found in image dir of tfrecords,this dataset is abandoned.')
            continue

        if not os.path.exists(dir_info):
            print(d, ' is not found in info dir of pickle,this dataset is abandoned.')
            continue

        lst_image.append([dir_image + '/' + filename_image for filename_image in os.listdir(dir_image)])
        lst_label.append([dir_label + '/' + filename_label for filename_label in os.listdir(dir_label)])
        lst_info.append([dir_info + '/' + filename_info for filename_info in os.listdir(dir_info)])
    dictionary = {'image': lst_image, 'label': lst_label, 'info': lst_info}
    if not os.path.exists(config['dir_list_tfrecord']):os.makedirs(config['dir_list_tfrecord'])

    pickle_filename = config['dir_list_tfrecord'] + '/' + filename_tfrec_pickle + '.pickle'
    if os.path.exists(pickle_filename):
        old_dict = pickle.load(open(pickle_filename, 'rb'))
        dictionary = {**old_dict, **dictionary}
    pickle.dump(dictionary, open( pickle_filename, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


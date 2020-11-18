import h5py
import os

class HD5F_Reader:

    def __init__(self, dataset, rootdir_file, padded=False):
        self.rootdir_file = rootdir_file
        self.dataset = dataset
        self.file = None
        self.img_IDs = []
        self.file_keys = []
        self.info_patient = {}
        self.read_HD5F_file(padded=False)

    def read_HD5F_file(self, padded=False):

        '''

        Reads HD5F file and returns keys of images and file
        Dataset used:
        This is designed for the Melanom Tumorvolume hdf5 data.
        The sample data shape is
            h5py {
                'image': (2, H, W, D), # (channels, height, width, depth)
                'mask': (1, H, W, D)    # (channels, height, width, depth)
                'mask_iso': (1, H, W, D)  # (channels, height, width, depth)
            }

        :param self.dataset: type str: Read the files from corresponding Dataset
        :param self.root_dir_img: type str: Directory, in which the files are stored
        :param padded: type bool: Choose between normal images or padded ones
        :return: data_image: type ndarray: numpy array of 3D data
        :return: info_patient: type dict:  Patient info
        '''

        ## this dataset contains 2 channels, PET, CT channel

        files_dir = list(file for file in os.listdir(self.rootdir_file))
        print(files_dir)

        file_images = None

        if padded:

            file_images = files_dir[2]  ## we take the case with padding
            print(file_images)

        else:

            file_images = files_dir[1]  ## we take the case without padding
            print(file_images)

        ## start reading the hd5f file
        try:
            self.file = h5py.File(self.rootdir_file + '/' + file_images, 'r')
            self.file_keys = list(self.file.keys())  # ('image', 'mask', 'mask_iso')
            print(self.file_keys)
            self.img_IDs = [image_key for image_key in self.file[self.file_keys[0]].keys()]
            print("Data could be read and loaded")


        except Exception as e:
            print("Data could not be read and loaded")


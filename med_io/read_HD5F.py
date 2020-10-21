import h5py
import os

def read_HD5F(config, dataset, root_dir_img):

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

    '''

    ## each dataset has keys 'image', 'mask', 'mask_iso'
    ## we take the file to the no padded dataset
    ##make a generator able to have the ids of the images
    # this will improve memory usage
    ## this dataset contains 2 channels, PET, CT channel

    files_dir = list(file for file in os.listdir(root_dir_img))
    print(files_dir)
    file_images = files_dir[1]  ## we take the case without padding
    print(file_images)

    ## start reading the hd5f file
    try:
        file = h5py.File(root_dir_img + '/' + file_images, 'r')
        file_keys = list(file.keys()) #('image', 'mask', 'mask_iso')
        print(file_keys)
        img_IDs = [image_key for image_key in file[file_keys[0]].keys()]
        print("Data could be read and loaded")


    except:
        print("Data could not be read and loaded")



    return img_IDs, file, file_keys

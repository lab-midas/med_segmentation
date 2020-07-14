import nibabel as nib
import skimage.io as io
import numpy as np

def read_nii_path(path=None):

    img=nib.load(path)
    img_arr=img.get_fdata()
    return np.array(img_arr)





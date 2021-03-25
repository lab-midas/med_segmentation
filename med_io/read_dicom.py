import pydicom
import numpy as np
import glob
import re
import SimpleITK as sitk


def read_dicom_dir(dim_dir=None, all_files=False, format=None, order='SliceLocation'):
    """
    Read dicom files from dir and export 3D data and patient information.
    :param dim_dir: type str: Directory to dicom files of one channel of one patient
    :param all_files: type bool: Read the files in dim_dir regardless of suffix name
    :param format: type str: Regular expression for choosing data files
    :param order: type str:  order of slice by reading the dicom files with package pydicom.
                             It can be chosen from 'SliceLocation', 'filename_last_8_character','NAKO'
    :return: data_image: type ndarray: numpy array of 3D data
    :return: info_patient: type dict:  Patient info

    """

    if not all_files:
        # read only files with suffix '.dcm'
        dicom_paths = glob.glob(dim_dir + '/*.dcm')
        dicom_paths = [dicom_path.replace('\\', '/') for dicom_path in dicom_paths]

    else:
        # read files with any suffix
        dicom_paths = glob.glob(dim_dir + '/*')
        dicom_paths = [dicom_path.replace('\\', '/') for dicom_path in dicom_paths]

    if format is not None:
        # filter the files with respect to regluation expression
        dicom_paths = [dicom_path for dicom_path in dicom_paths if re.match(dim_dir + '/' + format, dicom_path)]

    data_image = []
    info_patient = {}

    # sort the data files with respect to their slice location
    if order == 'SliceLocation':

        sorted(dicom_paths, key=lambda dicom_path: pydicom.dcmread(dicom_path).SliceLocation)

        # load data in each dicom path
        try:
            data_image = np.array([pydicom.dcmread(dicom_path).pixel_array for dicom_path in dicom_paths])
        except:
            print('load data by pydicom failed, trying simpleITK now...')
            try:
                data_image = read_dicom_dir_simple_ITK(dim_dir)
            except Exception as e:
                print('load data by pydicom and simpleITK failed')
                print('Error info:', e)

    elif order == 'filename_last_8_character':
        dicom_paths.sort(key=lambda x: int(x[-10:-4]))
        try:
            data_image = np.array([pydicom.dcmread(dicom_path).pixel_array for dicom_path in dicom_paths])
        except:
            print('load data by pydicom failed, trying simpleITK now...')
            try:
                data_image = read_dicom_dir_simple_ITK(dim_dir)
            except Exception as e:
                print('load data by pydicom and simpleITK failed')
                print('Error info:',e)

    elif order == 'NAKO':

        # Since NAKO images dataset mixes all channels in one scanning image,
        # all the channels will be split here in terms of their different dicom tags.
        # only can be loaded by pydicom
        datasets = [pydicom.dcmread(dp) for dp in dicom_paths]

        tags_0051_1019 = sorted(set([ds[0x51, 0x1019].value for ds in datasets]))  # tag ScanOptions in NAKO dicom
        tags_0018_0081 = sorted(set([ds[0x18, 0x81].value for ds in datasets]))  # tag EchoTime in NAKO dicom

        list0, tags = [], []
        for tag1 in tags_0051_1019:
            for tag2 in tags_0018_0081:
                list1 = [ds for ds in datasets if ds[0x51, 0x1019].value == tag1 and ds[0x18, 0x81].value == tag2]
                if list1 != []:
                    tags.append(str(tag1) + '_' + str(tag2))
                    list0.append(list1)  # store valid tag set

        info_patient['channels_images_NAKO'] = tags
        # list0 stores the independent channels
        for list2 in list0:
            list2 = sorted(list2, key=lambda ds: ds.SliceLocation)
            data_img = [ds.pixel_array for ds in list2]  # load data in each dicom path
            data_image.append(data_img)
        data_image = np.array(data_image)
    else:
        try:
            data_image = np.array([pydicom.dcmread(dicom_path).pixel_array for dicom_path in dicom_paths])
        except:
            print('load data by pydicom failed, trying simpleITK now...')
            try:
                data_image = read_dicom_dir_simple_ITK(dim_dir)
            except Exception as e:
                print('load data by pydicom and simpleITK failed')
                print('Error info:',e)

    # read info in dicom
    ds = pydicom.dcmread(dicom_paths[0])

    # create dict for storing patient info from dicom
    ds_key = dir(ds)

    # Remove the unnecessary info
    # d is the keys that are removed.
    d = ('PixelData')
    for item in dir(ds):
        if not item[0].isupper() or item in d:
            ds_key.remove(item)

    # write value in info_patient
    for k in ds_key:
        a = getattr(ds, k)
        # remove some info which is not supported for saving in pickle file
        if isinstance(a, pydicom.dataset.Dataset) or isinstance(a, pydicom.sequence.Sequence):
            continue
        info_patient[k] = getattr(ds, k)

    return data_image, info_patient


def read_dicom_dir_simple_ITK(dim_dir):
    """
    Alternative to pydicom, since package gdcm is not available in package pydicom.
    Especially for KORA dataset.
    :param dim_dir: type str: dir to dicom series
    :return: type ndarray: shape (z,y,x)
    """
    reader = sitk.ImageSeriesReader()
    seriesIDs = reader.GetGDCMSeriesIDs(dim_dir)
    dcm_series = reader.GetGDCMSeriesFileNames(dim_dir, seriesIDs[0])
    reader.SetFileNames(dcm_series)
    img = reader.Execute()
    result1=sitk.GetArrayFromImage(img)
    if result1.shape[0]>300: # if slice num>300
        result1=result1[::2,...] # downsamle the slice. because the shape of the kora doesn't match the label!!
    return result1

import pickle
from med_io.parser_tfrec import parser

from med_io.write_tf_record import write_tfrecord
import tensorflow as tf
import numpy as np
import os
from predict import *
import scipy.io as sio
from util import convert_integers_to_onehot, convert_onehot_to_integers
from plot.plot_config import *


class Patient:
    """
    Patient class to hold all patient and scan-specific parameters
    """
    def __init__(self):

        self.config = None
        self.AccessionNumber = None  # str
        self.AcquisitionDate = None  # str
        self.AcquisitionMatrix = None  # list
        self.AcquisitionNumber = None  # int
        self.AcquisitionTime = None  # str
        self.AngioFlag = None  # str
        self.BitsAllocated = None  # int
        self.BitsStored = None  # int
        self.BodyPartExamined = None  # str
        self.Columns = None  # int
        self.CommentsOnThePerformedProcedureStep = None  # str
        self.ContentDate = None  # str
        self.ContentTime = None  # str
        self.DateOfLastCalibration = None  # list
        self.DeviceSerialNumber = None  # str
        self.EchoNumbers = None  # int
        self.EchoTime = None  # float
        self.EchoTrainLength = None  # int
        self.FlipAngle = None  # float
        self.FrameOfReferenceUID = None  # str
        self.HighBit = None  # int
        self.ImageOrientationPatient = None  # list
        self.ImagePositionPatient = None  # list
        self.ImageType = None  # list
        self.ImagedNucleus = None  # str
        self.ImagingFrequency = None  # float
        self.InPlanePhaseEncodingDirection = None  # str
        self.InstanceCreationDate = None  # str
        self.InstanceCreationTime = None  # str
        self.InstanceNumber = None  # int
        self.InstitutionAddress = None  # str
        self.InstitutionName = None  # str
        self.InstitutionalDepartmentName = None  # str
        self.LargestImagePixelValue = None  # int
        self.MRAcquisitionType = None  # str
        self.MagneticFieldStrength = None  # float
        self.Manufacturer = None  # str
        self.ManufacturerModelName = None  # str
        self.Modality = None  # str
        self.NumberOfAverages = None  # float
        self.NumberOfPhaseEncodingSteps = None  # int
        self.PatientAge = None  # str
        self.PatientBirthDate = None  # str
        self.PatientBirthTime = None  # str
        self.PatientID = None  # str
        self.PatientName = None  # PersonName3
        self.PatientPosition = None  # str
        self.PatientSex = None  # str
        self.PatientSize = None  # float
        self.PatientWeight = None  # float
        self.PercentPhaseFieldOfView = None  # float
        self.PercentSampling = None  # float
        self.PerformedProcedureStepDescription = None  # str
        self.PerformedProcedureStepID = None  # str
        self.PerformedProcedureStepStartDate = None  # str
        self.PerformedProcedureStepStartTime = None  # str
        self.PerformingPhysicianName = None  # PersonName3
        self.PhotometricInterpretation = None  # str
        self.PixelBandwidth = None  # float
        self.PixelRepresentation = None  # int
        self.PixelSpacing = None  # list
        self.PositionReferenceIndicator = None  # str
        self.ProtocolName = None  # str
        self.ReferringPhysicianName = None  # PersonName3
        self.RepetitionTime = None  # float
        self.RequestedProcedureDescription = None  # str
        self.RequestingPhysician = None  # PersonName3
        self.Rows = None  # int
        self.SAR = None  # float
        self.SOPClassUID = None  # str
        self.SOPInstanceUID = None  # str
        self.SamplesPerPixel = None  # int
        self.ScanOptions = None  # str
        self.ScanningSequence = None  # str
        self.SequenceName = None  # str
        self.SequenceVariant = None  # list
        self.SeriesDate = None  # str
        self.SeriesDescription = None  # str
        self.SeriesInstanceUID = None  # str
        self.SeriesNumber = None  # int
        self.SeriesTime = None  # str
        self.SliceLocation = None  # float
        self.SliceThickness = None  # float
        self.SmallestImagePixelValue = None  # int
        self.SoftwareVersions = None  # str
        self.SpecificCharacterSet = None  # str
        self.StationName = None  # str
        self.StudyDate = None  # str
        self.StudyDescription = None  # str
        self.StudyID = None  # str
        self.StudyInstanceUID = None  # str
        self.StudyTime = None  # str
        self.TimeOfLastCalibration = None  # list
        self.TransmitCoilName = None  # str
        self.VariableFlipAngleFlag = None  # str
        self.WindowCenter = None  # float
        self.WindowCenterWidthExplanation = None  # str
        self.WindowWidth = None  # float
        self.name_ID = None
        self.tfrecord_name_ID_dir = None
        self.dataset = None

        # ----------------------------------------------------
        # keys that are in the mat.file (NAKO_AT, TULIP/PLIS)
        self.hip = None
        self.heartEnd = None
        self.wrist = None
        self.heel = None
        self.SpacingBetweenSlices = None

        self.tfrecords_info_path = None

    def initialize(self, config, name_ID, dataset):
        """
        Initialize this Patient class
        :param config: type dict, config parameter
        :param name_ID: type str, name ID of the patient
        :param dataset: type str, name of the dataset
        :return:  class Patient
        """

        self.config = config
        self.name_ID = name_ID
        self.dataset = dataset
        self.tfrecord_name_ID_dir = self.config['rootdir_tfrec'][self.dataset] + '/' + self.name_ID
        self.tfrecords_info_path = self.tfrecord_name_ID_dir + '/info/info.pickle'
        with open(self.tfrecords_info_path, 'rb') as fp:
            inf = pickle.load(fp)
        inf = inf['info_patient'][0]
        patient_keys = self.__dict__.keys()
        for info_key in inf.keys():
            if info_key in patient_keys:
                self.__dict__[info_key] = inf[info_key]
        return self

    def get_original_image(self):
        """
        Get this patient's original image.
        :return: type ndarray : data image
        """

        tf_data_path = self.tfrecord_name_ID_dir + '/image/image.tfrecords'
        return np.array(self.parse_single_tfrecords(tf_data_path)).astype(np.float32)

    def get_predict_image(self, img_type=int):
        """
        Get this patient's predict data by the trained model
        :param img_type: int, float, or uint8. type of the returned predict data
        :return: type ndarray: predict data
        """
        result_predict_img_dir = self.config['result_rootdir'] + '/' + self.config[
            'model'] + '/predict_result/' + self.dataset + '/' + self.name_ID
        if not os.path.exists(result_predict_img_dir) or os.listdir(result_predict_img_dir) == []:
            print('Predict result does not exists! Predicting now ...')
            return predict(self.config, [self.dataset],save_predict_data=True, name_ID=self.name_ID).astype(
                img_type)
        else:
            print('Predict result already exists. Loading now ...')
            result_predict_img_path = result_predict_img_dir + '/' + os.listdir(result_predict_img_dir)[0]
            return (sio.loadmat(result_predict_img_path)['predict_image']).astype(img_type)

    def get_label(self):
        """
        Get this patient's label
        :return: type ndarray: data label
        """
        tf_data_path = self.tfrecord_name_ID_dir + '/label/label.tfrecords'
        return np.array(self.parse_single_tfrecords(tf_data_path)).astype(np.float32)

    def get_plot(self, plot_function_name=None):
        """
        Get this patient's plot, saved in the specified dirs.
        :param plot_function_name: if specified figure function name is used. None if use the functions in config['plot_figure']
        :return:
        """
        label_onehot = self.get_label()
        img_data = self.get_original_image()
        predict_category = self.get_predict_image()
        predict_onehot = convert_integers_to_onehot(predict_category, num_classes=label_onehot.shape[-1])
        label_category = convert_onehot_to_integers(label_onehot)

        dict_data = {'predict_category': predict_category,
                     'predict_onehot': predict_onehot,
                     'label_category': label_category,
                     'label_onehot': label_onehot,
                     'original_image': img_data}

        if plot_function_name is not None:

            self.config['plot_figure'] = plot_function_name
            if not isinstance(self.config['plot_figure'], list):
                self.config['plot_figure'] = [self.config['plot_figure']]

        save_plot_rootdir = './Patient'
        plot_figures_single(self.config, dict_data, dataset=self.dataset, name_ID=self.name_ID,
                            client_save_rootdir=save_plot_rootdir)
        list_images_series = {'predict': [predict_onehot], 'label': [label_onehot]}
        plot_figures_dataset(self.config, list_images_series, dataset=self.dataset,
                             client_save_rootdir=save_plot_rootdir)

    def parse_single_tfrecords(self, tf_datapath):
        """
        parse this patient's tfrecords path to data
        :param tf_datapath: type str: path of tfrecords
        :return: type ndarray: image data from the tfrecords.
        """

        image_TFRecordDataset = tf.data.TFRecordDataset([tf_datapath])
        dataset = image_TFRecordDataset.map(parser)
        # Get the image data from tfrecords
        img_data = [elem[0].numpy() for elem in dataset][0]
        return img_data

    def set_original_image(self, new_data):
        """
        Modify the image dataset
        :param new_data: type 4 dimensional ndarray: custom dataset
        :return:
        """
        tf_data_path = self.tfrecord_name_ID_dir + '/image/image.tfrecords'
        write_tfrecord(new_data, tf_data_path)

    def set_label(self, new_data):
        """
        Modify the label dataset
        :param new_data: type 4 dimensional ndarray: custom dataset
        :return:
        """
        tf_data_path = self.tfrecord_name_ID_dir + '/label/label.tfrecords'
        write_tfrecord(new_data, tf_data_path)

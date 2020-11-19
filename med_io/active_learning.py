import tensorflow as tf
from med_io.parser_tfrec import parser
from med_io.get_pad_and_patch import get_fixed_patches_index
"""
Active learning parts for pipeline
"""
def query_training_patches(config, dataset_image_path, model):
    # Data loading inspired by predict.py

    # Reformat data path list: [[path1],[path2], ...] ->[[path1, path2, ...]]
    data_path_image_list = [t[i] for t in dataset_image_path for i in range(len(dataset_image_path[0]))]
    list_image_TFRecordDataset = [tf.data.TFRecordDataset(i) for i in data_path_image_list]

    for index, image_TFRecordDataset in enumerate(list_image_TFRecordDataset):
        dataset_image = image_TFRecordDataset.map(parser)
        img_data = [elem[0].numpy() for elem in dataset_image][0]
        
#         # bis hier aus predict.py ab hier aus pipeline
#         # patch the image
#         patch_size = config['patch_size']
#         dim = len(patch_size)
#         max_data_size = [config['max_shape']['image'][i] for i in range(dim)]

    # predict data-patches

    # calculate value of the patches for training

    # select the best n for training

    return 'hier liste mit ausgewählten patches?'


def uncertainty_sampling(prediction, computation='entropy'):
    """
        Calculate an estimation of uncertainty for the prediction and return
        this value. The prediction data must be a Tensor with a probability
        value for every class for every voxel.
        :parm prediction: type tf.Tensor, 4D prediction data where the first 3
        Dimensions are Space and the 4th the class probabilities
        :return: uncertainty_value
    """
    # Calculate an uncertainty value for every pixel producing an uncertainty-field
    if computation == 'entropy':
        # calculate the Shannon Entropy for every pixel as uncertainty value
        probs_log = tf.math.log(prediction)
        weighted_probs_log = tf.math.multiply(prediction, probs_log)
        uncertainty_field = tf.math.reduce_sum(weighted_probs_log, axis=-1)
    elif computation == 'least_confident':
        # pick the probability of the most likely class as uncertainty value
        uncertainty_field = tf.reduce_max(prediction, axis=-1)
    else:
        raise Exception('Unknown way of computing the uncertainty')

    # Average the values to get an average uncertainty for the entire prediction
    uncertainty = tf.math.reduce_mean(uncertainty_field)
    return uncertainty

import tensorflow as tf
"""
Active learning parts for pipeline
"""


def sort_by_informativeness(dataset, config=None):
    def map_informativeness(*args):
        """
        Map function for dataset, calculates an estimation of uncertainty for every
        patch (element of dataset) and adds this value to the dataset.
        :parm args: args[0] img_data (together with index) args[1] label_data
        :return:
        """
        if config['feed_pos']:
            patchs_imgs, _ = args[0]
        else:
            patchs_imgs = args[0]

        informativeness = 0
        return args[0], args[1], informativeness

    dataset = dataset.map(map_informativeness)
    # sort dataset by informativeness
    return dataset


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

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
        Map function for dataset, calculates an estimation of uncertainty for every
        patch (element of dataset) and adds this value to the dataset.
        :parm prediction: type tf.Tensor? prediction data of the patch, with
        :return: uncertainty_value
    """
#   print(prediction.shape())
    # from the prediction separate the predicted probabilities for each pixel by
    # class, then average the values to get an average probability for each class
    mean_class_probs = tf.math.reduce_mean(prediction, axis=(0, 1, 2))

#     class_maps = tf.unstack(prediction, axis=-1)
#     mean_class_probabilities = tf.constant([])
#     for class_map in class_maps:
#         mean_class_probs = tf.concat([tf.math.reduce_mean(class_map),
#                                               mean_class_probs])

    if computation == 'entropy':
        mean_class_probs_log = tf.math.log(mean_class_probs)
        entropy = tf.math.multiply(mean_class_probs, mean_class_probs_log)
        uncertainty = tf.math.reduce_sum(entropy)
    elif computation == 'least_confident':
        uncertainty = tf.reduce_max(mean_class_probs)
    else:
        raise Exception('Unknown way of computing the uncertainty')
    return uncertainty

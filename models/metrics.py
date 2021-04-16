import tensorflow as tf
import tensorflow.keras.backend as K


class Metric:
    """
    Extension of evaluation metrics not yet existing in keras and/or Tensorflow
    """

    """
    per class metrics
    """
    # sensitivity, recall, hit rate, true positive rate
    # TPR = TP/P = TP/(TP+FN) = 1-FNR
    def recall_per_class(self, selected_class, y_true, y_pred, config):
        smooth = 1e-16
        print('line17 metrics',y_true.shape, y_pred.shape)
        y_pred = tf.one_hot(tf.argmax(y_pred, axis=-1), config['channel_label_num'])
        print('selected_class',selected_class)
        true_positive = K.sum((y_true[..., selected_class] * y_pred[..., selected_class]))
        return (true_positive + smooth) / (K.sum(y_true[..., selected_class]) + smooth)

    # precision, positive predictive value (PPV)
    # PPV = TP/(TP+FP) = 1-FDR
    def precision_per_class(self, selected_class, y_true, y_pred, config):
        smooth = 1e-16
        y_pred = tf.one_hot(tf.argmax(y_pred, axis=-1), config['channel_label_num'])
        true_positive = K.sum((y_true[..., selected_class] * y_pred[..., selected_class]))
        return (true_positive + smooth) / (K.sum(y_pred[..., selected_class]) + smooth)

    # specificity, selectivity, true negative rate (TNR)
    # TNR = TN/N = TN/(TN+FP) = 1-FPR
    def specificity_per_class(self, selected_class, y_true, y_pred, config):
        smooth = 1e-16
        y_pred = tf.one_hot(tf.argmax(y_pred, axis=-1), config['channel_label_num'])
        true_negative = K.sum((y_true[..., selected_class] - 1) * (y_pred[..., selected_class] - 1))
        return (true_negative + smooth) / (K.abs(K.sum(y_true[..., selected_class] - 1)) + smooth)

    # F1 score
    # F1 = 2* (PPV*TPR)/(PPV+TPR)
    def f1_score_per_class(self, selected_class, y_true, y_pred, config):
        smooth = 1e-16
        recall_func = getattr(self, 'recall_all')
        precision_func = getattr(self, 'precision_all')
        recall = recall_func(self, selected_class, y_true, y_pred, config)
        precision = precision_func(self, selected_class, y_true, y_pred, config)
        return (2 * recall * precision + smooth) / (recall + precision + smooth)

    def dice_coef_per_class(self, selected_class, y_true, y_pred, config):
        """ Dice coefficient for Melanoma network
                            y_true: true targets tensor.
                            y_pred: predictions tensor.
                            Dice calculation with smoothing to avoid division by zero
        """
        # smooth = 1E-16
        # assert y_true.shape == y_pred.shape
        smooth = K.epsilon()
        sum_metric, weight_sum = 0, 0

        y_t = y_true[..., selected_class]
        y_p = y_pred[..., selected_class]
        intersection = tf.math.reduce_sum(y_t * y_p) * config['loss_channel_weight'][selected_class]
        denominator = tf.math.reduce_sum(y_t) + tf.math.reduce_sum(y_p) + smooth
        dice_coef = (2. * intersection / denominator)

        #y_mean = sum_metric / weight_sum
        return dice_coef

    """
    one against-rest metrics
    """
    # sensitivity, recall, hit rate, true positive rate
    # TPR = TP/P = TP/(TP+FN) = 1-FNR
    def recall_all(self, y_true, y_pred, config):
        smooth = 1e-16
        y_pred = tf.one_hot(tf.argmax(y_pred, axis=-1), config['channel_label_num'])
        true_positive = K.sum(y_true * y_pred)
        return (true_positive + smooth) / (K.sum(y_true) + smooth)

    # precision, positive predictive value (PPV)
    # PPV = TP/(TP+FP) = 1-FDR
    def precision_all(self, y_true, y_pred, config):
        smooth = 1e-16
        y_pred = tf.one_hot(tf.argmax(y_pred, axis=-1), config['channel_label_num'])
        true_positive = K.sum(y_true * y_pred)
        return (true_positive + smooth) / (K.sum(y_pred) + smooth)

    # specificity, selectivity, true negative rate (TNR)
    # TNR = TN/N = TN/(TN+FP) = 1-FPR
    def specificity_all(self, y_true, y_pred, config):
        smooth = 1e-16
        y_pred = tf.one_hot(tf.argmax(y_pred, axis=-1), config['channel_label_num'])
        true_negative = K.sum((y_true - 1) * (y_pred - 1))
        return (true_negative + smooth) / (K.abs(K.sum(y_true - 1)) + smooth)

    # F1 score
    # F1 = 2* (PPV*TPR)/(PPV+TPR)
    def f1_score_all(self, y_true, y_pred, config):
        smooth = 1e-16
        recall_func = getattr(self, 'recall_all')
        precision_func = getattr(self, 'precision_all')
        recall, precision = recall_func(self, y_true, y_pred, config), precision_func(self, y_true, y_pred, config)
        return (2 * recall * precision + smooth) / (recall + precision + smooth)

    def dice_coef_all(self, y_true, y_pred, config):
        """ Dice coefficient for Melanoma network
                            y_true: true targets tensor.
                            y_pred: predictions tensor.
                            Dice calculation with smoothing to avoid division by zero
        """
        # smooth = 1E-16
        # assert y_true.shape == y_pred.shape
        smooth = K.epsilon()
        #assert len(y_true.shape) == 5
        sum_metric, weight_sum = 0, 0

        for class_index in range(config['num_classes']):
            y_t = y_true[..., class_index]
            y_p = y_pred[..., class_index]
            intersection = tf.math.reduce_sum(y_t * y_p) * config['loss_channel_weight'][class_index]
            denominator = tf.math.reduce_sum(y_t) + tf.math.reduce_sum(y_p) + smooth

            metric = (2. * intersection / denominator)

            sum_metric += metric  ## this returns a tensor
            weight_sum += config['loss_channel_weight'][class_index]  ## this returns a tensor too

        y_mean = sum_metric / weight_sum

        return y_mean


def get_custom_metrics(amount_classes, name_metric, config):
    """
    Get list of metric functions by their name, and amount of class
    :param amount_classes: type int: amount of channel
    :param name_metric: type string: name of the metric
    :param config: type dict: config parameter.
    :return: list_metric: type list of function: list of metric funtions from class Metric()
    """

    metric_func = getattr(Metric, name_metric)
    list_metric = []
    if '_per_class' in name_metric:
        metric_func_per_class = lambda c: lambda y_true, y_pred: metric_func(Metric, c, y_true, y_pred, config)
        list_metric = [metric_func_per_class(c) for c in range(amount_classes)]
        for j, f in enumerate(list_metric):
            f.__name__ = name_metric + '_channel_' + str(j)
    if '_all' in name_metric:
        metric_func_all = lambda y_true, y_pred: metric_func(Metric, y_true, y_pred, config)
        metric_func_all.__name__ = name_metric
        list_metric = [metric_func_all]

    return list_metric

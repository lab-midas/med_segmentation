import tensorflow as tf
from keras.utils.np_utils import to_categorical

def test_tf_Argmax():
    y_true_b = tf.constant([[[[0,0,0,0,0], [0,1,1,0,1], [0,1,1,0,1],[1,0,0,1,1]],
                           [[0,0,0,0,0], [0,1,1,0,1], [0,1,1,0,1],[1,0,0,1,1]],
                           [[0,0,0,0,1], [0,1,1,0,1], [0,1,1,0,1],[1,0,0,1,1]]]])


    print("before rollaxis: ", y_true_b.shape)
    
    y_true_roll = tf.transpose(y_true_b, perm=[1, 2, 3, 0])
    print("after rollaxis: ", y_true_roll.shape)
    
    y_true_batch = tf.expand_dims(y_true_roll, axis=0)

    #y_true_max = tf.math.argmax(y_true, axis=-1)

    y_true_2 = to_categorical(y_true_batch, num_classes=2)
    y_true_one_hot = tf.convert_to_tensor(y_true_2, dtype=y_true_batch.dtype)
    #y_true_2 = tf.one_hot(y_true_batch, 2)
    print("one hot: ", y_true_one_hot.shape)

test_tf_Argmax()
print("ready")
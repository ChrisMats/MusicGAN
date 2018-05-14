import tensorflow as tf


# Helping functions

# import data


# Deconvolution Opperation
def conv1d_transpose(inputs,filters,kernel_width=25,stride=4,activation=tf.nn.relu,batchnorm = False,is_training=True,scope=None):

    deconv = tf.layers.conv2d_transpose(
        tf.expand_dims(inputs, axis=1),
        filters,
        (1, kernel_width),
        strides=(1, stride),
        padding='same',
        name=scope)[:, 0]

    if batchnorm:
        BN = tf.layers.batch_normalization(deconv, training=is_training, name=scope+"_BN")
        out = activation(BN, name=scope+"_ReLu")
    else:
        out = activation(deconv, name=scope+"_actv")
    return out

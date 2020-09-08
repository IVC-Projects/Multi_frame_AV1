import tensorflow as tf
import numpy as np


def resblock(temp_tensor, convId, weights):
    out_tensor = None
    skip_tensor = None
    conv_secondID = 0

    skip_tensor = temp_tensor

    # Conv, 1x1, filters=192 ,+ ReLU
    conv_w = tf.get_variable("conv_%02d_%02d_w" % (convId, conv_secondID), [1, 1, 96, 192],
                             initializer=tf.contrib.layers.xavier_initializer())
    conv_b = tf.get_variable("conv_%02d_%02d_b" % (convId, conv_secondID), [192],
                             initializer=tf.constant_initializer(0))
    weights.append(conv_w)
    weights.append(conv_b)
    out_tensor = tf.nn.relu(
        tf.nn.bias_add(tf.nn.conv2d(temp_tensor, conv_w, strides=[1, 1, 1, 1], padding='SAME'), conv_b))
    conv_secondID += 1

    # Conv, 1x1, filters=25
    conv_w = tf.get_variable("conv_%02d_%02d_w" % (convId, conv_secondID), [1, 1, 192, 25],
                             initializer=tf.contrib.layers.xavier_initializer())
    conv_b = tf.get_variable("conv_%02d_%02d_b" % (convId, conv_secondID), [25], initializer=tf.constant_initializer(0))
    weights.append(conv_w)
    weights.append(conv_b)
    out_tensor = tf.nn.bias_add(tf.nn.conv2d(out_tensor, conv_w, strides=[1, 1, 1, 1], padding='SAME'), conv_b)
    conv_secondID += 1

    # Conv, 3x3, filters=32
    conv_w = tf.get_variable("conv_%02d_%02d_w" % (convId, conv_secondID), [3, 3, 25, 96],
                             initializer=tf.contrib.layers.xavier_initializer())
    conv_b = tf.get_variable("conv_%02d_%02d_b" % (convId, conv_secondID), [96], initializer=tf.constant_initializer(0))
    weights.append(conv_w)
    weights.append(conv_b)
    out_tensor = tf.nn.bias_add(tf.nn.conv2d(out_tensor, conv_w, strides=[1, 1, 1, 1], padding='SAME'), conv_b)
    conv_secondID += 1

    # skip + out_tensor
    out_tensor = tf.add(skip_tensor, out_tensor)

    return out_tensor


def network(frame1, frame2, frame3):
    weights = []
    input_tensor_Concat = None
    convId = 0
    conv_00_w = tf.get_variable("conv_%02d_w" % (convId), [3, 3, 1, 32],
                                initializer=tf.contrib.layers.xavier_initializer())
    conv_00_b = tf.get_variable("conv_%02d_b" % (convId), [32], initializer=tf.constant_initializer(0))
    weights.append(conv_00_w)
    weights.append(conv_00_b)

    tensor1 = tf.nn.bias_add(tf.nn.conv2d(frame1, conv_00_w, strides=[1, 1, 1, 1], padding='SAME'), conv_00_b)
    tensor2 = tf.nn.bias_add(tf.nn.conv2d(frame2, conv_00_w, strides=[1, 1, 1, 1], padding='SAME'), conv_00_b)
    tensor3 = tf.nn.bias_add(tf.nn.conv2d(frame3, conv_00_w, strides=[1, 1, 1, 1], padding='SAME'), conv_00_b)

    input_tensor_Concat = tf.concat([tensor1, tensor2, tensor3], axis=3)

    convId += 1

    # Residual Block x 8
    for i in range(8):
        input_tensor_Concat = resblock(input_tensor_Concat, convId, weights)
        convId += 1

    conv_w = tf.get_variable("conv_%02d_w" % (convId), [3, 3, 96, 1],
                             initializer=tf.contrib.layers.xavier_initializer())
    conv_b = tf.get_variable("conv_%02d_b" % (convId), [1], initializer=tf.constant_initializer(0))
    weights.append(conv_w)
    weights.append(conv_b)
    input_tensor_Concat = tf.nn.bias_add(tf.nn.conv2d(input_tensor_Concat, conv_w, strides=[1, 1, 1, 1], padding='SAME'), conv_b)

    input_tensor_Concat = tf.add(input_tensor_Concat, frame2)

    return input_tensor_Concat
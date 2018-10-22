import tensorflow as tf
import numpy as np


def squash(inputs, axis):

    norm = tf.norm(inputs, axis=axis, keep_dims=True)
    squared_norm = tf.square(norm)

    return squared_norm / (1. + squared_norm) / norm * inputs


def element_wise_matmul(a, b):

    l, m = a.shape.as_list()[-2:]
    m, n = b.shape.as_list()[-2:]

    s = a.shape.as_list()
    d = len(s)

    a = tf.tile(a, [1] * (d - 1) + [n])
    a = tf.reshape(a, [-1] + s[1:-2] + [l * n, m])

    b = tf.tile(b, [1] * (d - 1) + [l])
    b = tf.transpose(b, list(range(d - 2)) + [d - 1, d - 2])

    c = a * b
    c = tf.reduce_sum(c, axis=-1)
    c = tf.reshape(c, [-1] + s[1:-2] + [l, n])

    return c


def capsule_dense(inputs, units, dims, num_iters=3, name="capsule_dense", reuse=None):

    with tf.variable_scope(name, reuse=reuse):

        in_units = inputs.shape[1]
        in_dims = inputs.shape[2]

        weights = tf.get_variable(
            name="weights",
            shape=[in_units, units, in_dims, dims],
            dtype=tf.float32,
            initializer=tf.variance_scaling_initializer(
                scale=2.0,
                mode="fan_in",
                distribution="normal",
            ),
            trainable=True
        )

        biasses = tf.get_variable(
            name="biasses",
            shape=[units, dims],
            initializer=tf.zeros_initializer(),
            trainable=True
        )

        # -> inputs: [batch_size, in_units, in_dims]
        inputs = tf.tile(inputs, [1, units, 1])
        # -> inputs: [batch_size, in_units * out_units, in_dims]
        inputs = tf.reshape(inputs, [-1, in_units, units, in_dims])
        # -> inputs: [batch_size, in_units, out_units, in_dims]
        inputs = tf.expand_dims(inputs, axis=-2)
        # -> inputs: [batch_size, in_units, out_units, 1, in_dims]

        # -> weights: [in_units, out_units, in_dims, out_dims]
        weights = tf.expand_dims(weights, axis=0)
        # -> weights: [1, in_units, out_units, in_dims, out_dims]

        # -> inputs: [batch_size, in_units, out_units, 1, in_dims]
        inputs = element_wise_matmul(inputs, weights)
        # -> inputs: [batch_size, in_units, out_units, 1, out_dims]
        inputs = tf.squeeze(inputs, axis=-2)
        # -> inputs: [batch_size, in_units, out_units, out_dims]

        inputs_fixed = tf.stop_gradient(inputs)

        logits = tf.constant(
            value=0.0,
            dtype=tf.float32,
            shape=[in_units, units]
        )

        for i in range(num_iters):

            softmax = tf.nn.softmax(logits, dim=0)
            # -> softmax: [in_units, out_units]
            softmax = tf.expand_dims(softmax, axis=-1)
            # -> softmax: [in_units, out_units, 1]

            outputs = inputs_fixed * softmax
            # -> outputs: [batch_size, in_units, out_units, out_dims]
            outputs = tf.reduce_sum(outputs, axis=1, keep_dims=True)
            # -> outputs: [batch_size, 1, out_units, out_dims]
            outputs += biasses
            # -> outputs: [batch_size, 1, out_units, out_dims]
            outputs = squash(outputs, axis=-1)
            # -> outputs: [batch_size, 1, out_units, out_dims]

            similarity = inputs_fixed * outputs
            # -> similarity: [batch_size, in_units, out_units, out_dims]
            similarity = tf.reduce_sum(similarity, axis=-1)
            # -> similarity: [batch_size, in_units, out_units]
            similarity = tf.reduce_mean(similarity, axis=0)
            # -> similarity: [in_units, out_units]
            logits += similarity

        else:

            softmax = tf.nn.softmax(logits, axis=0)
            # -> softmax: [in_units, out_units]
            softmax = tf.expand_dims(softmax, axis=-1)
            # -> softmax: [in_units, out_units, 1]

            outputs = inputs * softmax
            # -> outputs: [batch_size, in_units, out_units, out_dims]
            outputs = tf.reduce_sum(outputs, axis=1, keep_dims=True)
            # -> outputs: [batch_size, 1, out_units, out_dims]
            outputs += biasses
            # -> outputs: [batch_size, 1, out_units, out_dims]
            outputs = squash(outputs, axis=-1)
            # -> outputs: [batch_size, 1, out_units, out_dims]
            outputs = tf.squeeze(outputs, axis=1)

        return outputs

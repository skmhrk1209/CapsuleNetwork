import tensorflow as tf
import numpy as np


def squash(inputs, axis):

    norm = tf.norm(inputs, axis=axis, keep_dims=True)
    squared_norm = tf.square(norm)

    return squared_norm / (1. + squared_norm) / norm * inputs


def capsule_dense(inputs, in_units, out_units, in_dims, out_dims, num_routing=3, name="capsule_dense", reuse=None):

    with tf.variable_scope(name, reuse=reuse):

        weights = tf.get_variable(
            name="weights",
            shape=[in_units, in_dims, out_units * out_dims],
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
            shape=[out_units, out_dims],
            initializer=tf.zeros_initializer(),
            trainable=True
        )

        # -> inputs: [batch_size, in_units, in_dims]
        inputs = tf.expand_dims(inputs, -1)
        # -> inputs: [batch_size, in_units, in_dims, 1]
        inputs = tf.tile(inputs, [1, 1, 1, out_units * out_dims])
        # -> inputs: [batch_size, in_units, in_dims, out_units * out_dims]
        inputs = tf.reduce_sum(inputs * weights, axis=2)
        # -> inputs: [batch_size, in_units, out_units * out_dims]
        inputs = tf.reshape(inputs, [-1, in_units, out_units, out_dims])
        # -> inputs: [batch_size, in_units, out_units, out_dims]

        logits = tf.fill(
            dims=[in_units, out_units],
            value=0.0
        )

        for _ in range(num_routing - 1):

            softmax = tf.nn.softmax(logits, dim=0)
            # -> softmax: [in_units, out_units]
            softmax = tf.expand_dims(softmax, axis=0)
            # -> softmax: [1, in_units, out_units]
            softmax = tf.expand_dims(softmax, axis=-1)
            # -> softmax: [1, in_units, out_units, 1]

            outputs = inputs * softmax
            # -> outputs: [batch_size, in_units, out_units, out_dims]
            outputs = tf.reduce_sum(outputs, axis=1)
            # -> outputs: [batch_size, out_units, out_dims]
            outputs += biasses
            # -> outputs: [batch_size, out_units, out_dims]
            outputs = squash(outputs, axis=-1)
            # -> outputs: [batch_size, out_units, out_dims]

            outputs = tf.expand_dims(outputs, axis=1)
            # -> outputs: [batch_size, 1, out_units, out_dims]
            similarity = inputs * outputs
            # -> similarity: [batch_size, in_units, out_units, out_dims]
            similarity = tf.reduce_sum(similarity, axis=-1)
            # -> similarity: [batch_size, in_units, out_units]
            similarity = tf.reduce_mean(similarity, axis=0)
            # -> similarity: [in_units, out_units]

            logits += similarity

        else:

            softmax = tf.nn.softmax(logits, dim=0)
            # -> softmax: [in_units, out_units]
            softmax = tf.expand_dims(softmax, axis=0)
            # -> softmax: [1, in_units, out_units]
            softmax = tf.expand_dims(softmax, axis=-1)
            # -> softmax: [1, in_units, out_units, 1]

            outputs = inputs * softmax
            # -> outputs: [batch_size, in_units, out_units, out_dims]
            outputs = tf.reduce_sum(outputs, axis=1)
            # -> outputs: [batch_size, out_units, out_dims]
            outputs += biasses
            # -> outputs: [batch_size, out_units, out_dims]
            outputs = squash(outputs, axis=-1)
            # -> outputs: [batch_size, out_units, out_dims]

        return outputs

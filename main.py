import tensorflow as tf
import numpy as np
import argparse
import ops

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=10000, help="number of training steps")
parser.add_argument("--num_epochs", type=int, default=100, help="number of training epochs")
parser.add_argument("--batch_size", type=int, default=100, help="batch size")
parser.add_argument("--model_dir", type=str, default="mnist_capsule_network_model", help="model directory")
parser.add_argument('--train', action="store_true", help="with training")
parser.add_argument('--eval', action="store_true", help="with evaluation")
parser.add_argument('--predict', action="store_true", help="with prediction")
parser.add_argument('--gpu', type=str, default="0", help="gpu id")
args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)


def capsule_network_model_fn(features, labels, mode, params):

    inputs = features["images"]

    inputs = tf.layers.flatten(inputs)

    inputs = tf.expand_dims(inputs, axis=-1)

    inputs = ops.capsule_dense(
        inputs=inputs,
        in_units=784,
        out_units=128,
        in_dims=1,
        out_dims=8,
        name="capsule_dense_0"
    )

    inputs = ops.capsule_dense(
        inputs=inputs,
        in_units=128,
        out_units=128,
        in_dims=8,
        out_dims=16,
        name="capsule_dense_1"
    )

    inputs = ops.capsule_dense(
        inputs=inputs,
        in_units=128,
        out_units=10,
        in_dims=16,
        out_dims=32,
        name="capsule_dense_2"
    )

    logits = tf.norm(inputs, axis=-1)
    classes = tf.argmax(logits, axis=-1)

    accuracy = tf.metrics.accuracy(labels, classes)
    accuracy_value = tf.identity(accuracy[0], "accuracy_value")

    labels = tf.one_hot(labels, 10)
    loss = labels * tf.square(tf.maximum(0.0, 0.9 - logits)) + (1 - labels) * tf.square(tf.maximum(0.0, logits - 0.1)) * 0.5
    loss = tf.reduce_mean(tf.reduce_sum(loss, axis=-1))

    if mode == tf.estimator.ModeKeys.TRAIN:

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):

            train_op = tf.train.AdamOptimizer().minimize(
                loss=loss,
                global_step=tf.train.get_global_step()
            )

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op
        )

    if mode == tf.estimator.ModeKeys.EVAL:

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops={"accuracy": accuracy}
        )


def main(unused_argv):

    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_images = mnist.train.images.astype(np.float32)
    test_images = mnist.test.images.astype(np.float32)
    train_labels = mnist.train.labels.astype(np.int32)
    test_labels = mnist.test.labels.astype(np.int32)

    mnist_classifier = tf.estimator.Estimator(
        model_fn=capsule_network_model_fn,
        model_dir=args.model_dir,
        config=tf.estimator.RunConfig().replace(
            session_config=tf.ConfigProto(
                gpu_options=tf.GPUOptions(
                    visible_device_list=args.gpu,
                    allow_growth=True
                )
            )
        )
    )

    if args.train:

        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"images": train_images},
            y=train_labels,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            shuffle=True
        )

        mnist_classifier.train(
            input_fn=train_input_fn,
            hooks=[
                tf.train.LoggingTensorHook(
                    tensors={"accuracy": "accuracy_value"},
                    every_n_iter=100
                )
            ]
        )

    if args.eval:

        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"images": test_images},
            y=test_labels,
            batch_size=args.batch_size,
            num_epochs=1,
            shuffle=False
        )

        eval_results = mnist_classifier.evaluate(
            input_fn=eval_input_fn
        )

        print(eval_results)


if __name__ == "__main__":
    tf.app.run()

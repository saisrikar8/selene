import tensorflow as tf
import keras


@keras.saving.register_keras_serializable()
class Mish(tf.keras.layers.Layer):
    def call(self, inputs):
        return inputs * tf.math.tanh(tf.math.softplus(inputs))
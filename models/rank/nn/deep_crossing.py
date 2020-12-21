import tensorflow as tf
from tensorflow.python.feature_column.feature_column_lib import DenseColumn
from tensorflow.python.feature_column.feature_column_v2 import FeatureTransformationCache, _StateManagerImplV2
from typing import List


class DeepCrossingModel(tf.keras.Model):
    def __init__(self, output_dim: int, feature_columns: List[DenseColumn], residual_units: List[int], name=None,
                 **kwargs):
        super(DeepCrossingModel, self).__init__(name=name, **kwargs)
        self.output_dim = output_dim
        self.feature_columns = sorted(feature_columns, key=lambda column: column.name)
        self.residual_units = residual_units
        self.stack_layer = tf.keras.layers.Lambda(lambda tensors: tf.concat(tensors, axis=1), name="StackLayer")
        self.residual_layer = ResidualLayer(hidden_units=residual_units)
        self.score_layer = tf.keras.layers.Dense(units=output_dim, name="ScoreLayer")
        self.state_manager = _StateManagerImplV2(self, trainable=self.trainable)

    def get_config(self):
        from tensorflow.python.feature_column.feature_column_lib import serialize_feature_columns
        config = {
            "output_dim": self.output_dim,
            "feature_columns": serialize_feature_columns(self.feature_columns),
            "residual_units": tf.keras.utils.serialize_keras_object(self.residual_units)
        }
        base_config = super(DeepCrossingModel, self).get_config()
        return {**base_config, **config}

    def build(self, input_shape):
        for column in self.feature_columns:
            column.create_state(self.state_manager)

    def call(self, inputs, training=None, mask=None):
        transformation_cache = FeatureTransformationCache(inputs)
        input_tensors = []
        for column in self.feature_columns:
            dense_tensor = column.get_dense_tensor(transformation_cache, self.state_manager)
            input_tensors.append(dense_tensor)
        tensor = self.stack_layer(input_tensors)
        tensor = self.residual_layer(tensor)
        logits = self.score_layer(tensor)
        return logits


class ResidualLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_units: List[int], trainable=True, name=None, **kwargs):
        super(ResidualLayer, self).__init__(trainable=trainable, name=name, **kwargs)
        self.dense_layers = []
        self.hidden_units = hidden_units
        for hidden_unit in hidden_units:
            dense_layer = tf.keras.layers.Dense(units=hidden_unit, activation=tf.keras.activations.relu)
            self.dense_layers.append(dense_layer)
        self.last_dense_layer = None

    def get_config(self):
        config = {
            "hidden_units": tf.keras.utils.serialize_keras_object(self.hidden_units)
        }
        base_config = super(ResidualLayer, self).get_config()
        return {**base_config, **config}

    def build(self, input_shape):
        input_dim = tf.TensorShape(input_shape)[1]
        self.last_dense_layer = tf.keras.layers.Dense(units=input_dim, activation=None)

    def call(self, inputs, **kwargs):
        net = inputs
        for dense_layer in self.dense_layers:
            net = dense_layer(net)
        net = self.last_dense_layer(net)
        return tf.keras.activations.relu(inputs + net)

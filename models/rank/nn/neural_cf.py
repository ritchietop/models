import tensorflow as tf
from tensorflow.python.feature_column.feature_column_lib import DenseColumn
from tensorflow.python.feature_column.feature_column_v2 import FeatureTransformationCache, _StateManagerImplV2
from typing import List


class NeuralCfModel(tf.keras.Model):
    def __init__(self, output_dim: int, user_columns: List[DenseColumn], item_columns: List[DenseColumn],
                 hidden_units: List[int], dropout: float, name=None, **kwargs):
        super(NeuralCfModel, self).__init__(name=name, **kwargs)
        self.output_dim = output_dim
        self.user_columns = sorted(user_columns, key=lambda column: column.name)
        self.item_columns = sorted(item_columns, key=lambda column: column.name)
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.state_manager = _StateManagerImplV2(self, trainable=self.trainable)
        self.gmf_layer = tf.keras.layers.Lambda(lambda inputs: tf.math.multiply(inputs[0], inputs[1]), name="GMFLayer")
        self.denser_layers = []
        self.dropout_layers = []
        for hidden_unit in self.hidden_units:
            dense_layer = tf.keras.layers.Dense(units=hidden_unit, activation=tf.keras.activations.relu)
            self.denser_layers.append(dense_layer)
            if self.dropout > 0:
                dropout_layer = tf.keras.layers.Dropout(rate=dropout)
                self.dropout_layers.append(dropout_layer)
        self.score_layer = tf.keras.layers.Dense(units=output_dim, activation=None, name="ScoreLayer")

    def get_config(self):
        from tensorflow.python.feature_column.feature_column_lib import serialize_feature_columns
        config = {
            "output_dim": self.output_dim,
            "user_columns": serialize_feature_columns(self.user_columns),
            "item_columns": serialize_feature_columns(self.item_columns),
            "hidden_units": tf.keras.utils.serialize_keras_object(self.hidden_units),
            "dropout": self.dropout
        }
        base_config = super(NeuralCfModel, self).get_config()
        return {**base_config, **config}

    def build(self, input_shape):
        user_dim = item_dim = 0
        for user_column in self.user_columns:
            user_column.create_state(self.state_manager)
            user_dim += user_column.variable_shape.num_elements()
        for item_column in self.item_columns:
            item_column.create_state(self.state_manager)
            item_dim += item_column.variable_shape.num_elements()
        if user_dim != item_dim:
            raise ValueError("user input dim must be same with item input dim.")

    def call(self, inputs, training=None, mask=None):
        transformation_cache = FeatureTransformationCache(inputs)
        user_tensors = []
        for user_column in self.user_columns:
            user_tensor = user_column.get_dense_tensor(transformation_cache, self.state_manager)
            user_tensors.append(user_tensor)
        item_tensors = []
        for item_column in self.item_columns:
            item_tensor = item_column.get_dense_tensor(transformation_cache, self.state_manager)
            item_tensors.append(item_tensor)
        user_input_tensor = tf.concat(user_tensors, axis=1)
        item_input_tensor = tf.concat(item_tensors, axis=1)
        gmf_tensor = self.gmf_layer([user_input_tensor, item_input_tensor])
        mlp_tensor = tf.concat([user_input_tensor, item_input_tensor], axis=1)
        for layer_index in range(len(self.denser_layers)):
            mlp_tensor = self.denser_layers[layer_index](mlp_tensor)
            if self.dropout_layers:
                mlp_tensor = self.dropout_layers[layer_index](mlp_tensor, training=True)
        output_tensor = tf.concat([gmf_tensor, mlp_tensor], axis=1)
        logits = self.score_layer(output_tensor)
        return logits

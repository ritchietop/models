import tensorflow as tf
from tensorflow.python.feature_column.feature_column_lib import CategoricalColumn, DenseColumn, FeatureColumn
from tensorflow.python.feature_column.feature_column_v2 import FeatureTransformationCache, _StateManagerImplV2
from typing import List


class WideDeepModel(tf.keras.Model):
    def __init__(self, units: int, linear_columns: List[FeatureColumn], dnn_columns: List[DenseColumn],
                 hidden_units: List[int], dropout: float, name=None, **kwargs):
        super(WideDeepModel, self).__init__(name=name, **kwargs)
        self.units = units
        self.linear_columns = sorted(linear_columns, key=lambda column: column.name)
        self.dnn_columns = sorted(dnn_columns, key=lambda column: column.name)
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.state_manager = _StateManagerImplV2(self, trainable=self.trainable)
        self.dense_layers = []
        self.dropout_layers = []
        for hidden_unit in hidden_units:
            dense_layer = tf.keras.layers.Dense(units=hidden_unit, activation=tf.keras.activations.relu)
            self.dense_layers.append(dense_layer)
            if dropout > 0:
                dropout_layer = tf.keras.layers.Dropout(rate=dropout)
                self.dropout_layers.append(dropout_layer)
        self.logits_layer = tf.keras.layers.Dense(units=units, activation=None)

    def get_config(self):
        from tensorflow.python.feature_column.feature_column_lib import serialize_feature_columns
        config = {
            "units": self.units,
            "linear_columns": serialize_feature_columns(self.linear_columns),
            "dnn_columns": serialize_feature_columns(self.dnn_columns),
            "hidden_units": tf.keras.utils.serialize_keras_object(self.hidden_units),
            "dropout": self.dropout
        }
        base_config = super(WideDeepModel, self).get_config()
        return {**base_config, **config}

    def build(self, input_shape):
        for column in self.linear_columns:
            column.create_state(self.state_manager)
            if isinstance(column, CategoricalColumn):
                num_elements = column.num_buckets
            elif isinstance(column, DenseColumn):
                num_elements = column.variable_shape.num_elements()
            else:
                raise ValueError("Column must be CategoricalColumn or DenseColumn. But get {}" % type(column))
            self.state_manager.create_variable(
                feature_column=column,
                name="linear_weights",
                shape=(num_elements, 1),
                trainable=self.trainable)
        for column in self.dnn_columns:
            column.create_state(self.state_manager)
            self.state_manager.create_variable(
                feature_column=column,
                name="dnn_weights",
                shape=(column.variable_shape.num_elements(), 1),
                dtype=self.dtype,
                trainable=True)

    def call(self, inputs, training=None, mask=None):
        transformation_cache = FeatureTransformationCache(inputs)
        linear_tensors = []
        for column in self.linear_columns:
            column_weights = self.state_manager.get_variable(column, "linear_weights")
            if isinstance(column, CategoricalColumn):
                sparse_tensors = column.get_sparse_tensors(transformation_cache, self.state_manager)
                linear_tensor = tf.nn.safe_embedding_lookup_sparse(
                    embedding_weights=column_weights,
                    sparse_ids=sparse_tensors.id_tensor,
                    sparse_weights=sparse_tensors.weight_tensor,
                    combiner="sum")
            elif isinstance(column, DenseColumn):
                dense_tensor = column.get_dense_tensor(transformation_cache, self.state_manager)
                linear_tensor = tf.matmul(dense_tensor, column_weights)
            linear_tensors.append(linear_tensor)
        linear_net = tf.concat(linear_tensors, axis=1)
        dnn_tensors = []
        for column in self.dnn_columns:
            column_weights = self.state_manager.get_variable(column, "dnn_weights")
            dnn_tensor = column.get_dense_tensor(transformation_cache, self.state_manager)
            dnn_tensor = tf.matmul(dnn_tensor, column_weights)
            dnn_tensors.append(dnn_tensor)
        dnn_net = tf.concat(dnn_tensors, axis=1)
        for layer_index in range(len(self.dense_layers)):
            dnn_net = self.dense_layers[layer_index](dnn_net)
            if self.dropout_layers:
                dnn_net = self.dropout_layers[layer_index](dnn_net, training=True)
        net = tf.concat([linear_net, dnn_net], axis=1)
        logits = self.logits_layer(net)
        return logits

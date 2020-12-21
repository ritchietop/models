import tensorflow as tf
from tensorflow.python.feature_column.feature_column_lib import FeatureColumn, DenseColumn, CategoricalColumn
from tensorflow.python.feature_column.feature_column_v2 import FeatureTransformationCache, _StateManagerImplV2
from typing import List


class PNNModel(tf.keras.Model):
    def __init__(self, output_dim: int, embedding_size: int, feature_columns: List[FeatureColumn], product_type: str,
                 hidden_units: List[int], dropout: float, name=None, **kwargs):
        super(PNNModel, self).__init__(name=name, **kwargs)
        self.output_dim = output_dim
        self.embedding_size = embedding_size
        self.feature_columns = sorted(feature_columns, key=lambda column: column.name)
        self.product_type = product_type
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.state_manager = _StateManagerImplV2(self, trainable=self.trainable)
        self.product_layer = None
        self.product_bias = None
        self.dense_layers = []
        self.dropout_layers = []
        for hidden_unit in hidden_units:
            dense_layer = tf.keras.layers.Dense(units=hidden_unit, activation=tf.keras.activations.relu)
            self.dense_layers.append(dense_layer)
            if dropout > 0:
                dropout_layer = tf.keras.layers.Dropout(rate=dropout)
                self.dropout_layers.append(dropout_layer)
        self.score_layer = tf.keras.layers.Dense(units=output_dim, activation=None)

    def get_config(self):
        from tensorflow.python.feature_column.feature_column_lib import serialize_feature_columns
        config = {
            "output_dim": self.output_dim,
            "embedding_size": self.embedding_size,
            "feature_columns": serialize_feature_columns(self.feature_columns),
            "product_type": self.product_type,
            "hidden_units": tf.keras.utils.serialize_keras_object(self.hidden_units),
            "dropout": self.dropout
        }
        base_config = super(PNNModel, self).get_config()
        return {**base_config, **config}

    def build(self, input_shape):
        for column in self.feature_columns:
            if isinstance(column, DenseColumn):
                num_elements = column.variable_shape.num_elements()
            elif isinstance(column, CategoricalColumn):
                num_elements = column.num_buckets
            else:
                raise ValueError("Only Support DenseColumn or CategoricalColumn. But get {}" % type(column))
            self.state_manager.create_variable(
                feature_column=column,
                name="embedding_weights",
                shape=(num_elements, self.embedding_size),
                dtype=self.dtype,
                trainable=True)
        self.product_bias = self.add_weight(
            name="product_bias",
            shape=(self.embedding_size,),
            dtype=self.dtype,
            trainable=True)

    def call(self, inputs, training=None, mask=None):
        transformation_cache = FeatureTransformationCache(inputs)
        embedding_tensors = []
        for column in self.feature_columns:
            embedding_weights = self.state_manager.get_variable(column, "embedding_weights")
            if isinstance(column, DenseColumn):
                dense_tensor = column.get_dense_tensor(transformation_cache, self.state_manager)
                input_tensor = tf.matmul(dense_tensor, embedding_weights)
            elif isinstance(column, CategoricalColumn):
                sparse_tensors = column.get_sparse_tensors(transformation_cache, self.state_manager)
                input_tensor = tf.nn.safe_embedding_lookup_sparse(
                    embedding_weights=embedding_weights,
                    sparse_ids=sparse_tensors.id_tensor,
                    sparse_weights=sparse_tensors.weight_tensor,
                    combiner="mean")
            embedding_tensors.append(input_tensor)
        product_tensor = self.product_layer(embedding_tensors)
        net = tf.add_n(embedding_tensors) + product_tensor
        net = tf.nn.bias_add(net, self.product_bias)
        net = tf.keras.activations.relu(net)
        for layer_index in range(len(self.dense_layers)):
            net = self.dense_layers[layer_index](net)
            if self.dropout_layers:
                net = self.dropout_layers[layer_index](net, training=True)
        logits = self.score_layer(net)
        return logits

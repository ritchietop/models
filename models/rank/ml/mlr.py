import tensorflow as tf
from tensorflow.python.feature_column.feature_column_lib import FeatureColumn, CategoricalColumn, DenseColumn
from tensorflow.python.feature_column.feature_column_v2 import FeatureTransformationCache, _StateManagerImplV2
from typing import List


class MLRModel(tf.keras.Model):
    def __init__(self, split_count: int, columns: List[FeatureColumn], l2_factor: float, name=None, **kwargs):
        super(MLRModel, self).__init__(name=name, **kwargs)
        self.split_count = split_count
        self.columns = columns
        self.l2_factor = l2_factor
        self.regularizer = tf.keras.regularizers.l2(l2_factor)
        self.state_manager = _StateManagerImplV2(self, trainable=True)
        self.softmax_bias = None
        self.sigmoid_bias = None

    def get_config(self):
        from tensorflow.python.feature_column.feature_column_lib import serialize_feature_columns
        config = {
            "split_count": self.split_count,
            "columns": serialize_feature_columns(self.columns),
            "l2_factor": self.l2_factor
        }
        base_config = super(MLRModel, self).get_config()
        return {**base_config, **config}

    def build(self, input_shape):
        for column in self.columns:
            column.create_state(self.state_manager)
            if isinstance(column, DenseColumn):
                first_dim = column.variable_shape.num_elements()
            elif isinstance(column, CategoricalColumn):
                first_dim = column.num_buckets
            else:
                raise ValueError("Only Support DenseColumn or CategoricalColumn")
            self.state_manager.create_variable(
                feature_column=column,
                name="softmax_weights",
                shape=(first_dim, self.split_count),
                dtype=self.dtype,
                trainable=True)
            self.state_manager.create_variable(
                feature_column=column,
                name="sigmoid_weights",
                shape=(first_dim, self.split_count),
                dtype=self.dtype,
                trainable=True)
        self.softmax_bias = self.add_weight(
            name="softmax_bias",
            shape=(self.split_count,),
            dtype=self.dtype,
            trainable=True)
        self.sigmoid_bias = self.add_weight(
            name="sigmoid_bias",
            shape=(self.split_count,),
            dtype=self.dtype,
            trainable=True)

    def call(self, inputs, training=None, mask=None):
        transformation_cache = FeatureTransformationCache(inputs)
        softmax_output_tensors = []
        sigmoid_output_tensors = []
        for column in self.columns:
            softmax_weights = self.state_manager.get_variable(column, "softmax_weights")
            self.add_loss(self.regularizer(softmax_weights))
            sigmoid_weights = self.state_manager.get_variable(column, "sigmoid_weights")
            self.add_loss(self.regularizer(sigmoid_weights))
            if isinstance(column, DenseColumn):
                dense_tensor = column.get_dense_tensor(transformation_cache, self.state_manager)
                softmax_output_tensor = tf.matmul(dense_tensor, softmax_weights)
                sigmoid_output_tensor = tf.matmul(dense_tensor, sigmoid_weights)
            elif isinstance(column, CategoricalColumn):
                sparse_tensors = column.get_sparse_tensors(transformation_cache, None)
                softmax_output_tensor = tf.nn.safe_embedding_lookup_sparse(
                    embedding_weights=softmax_weights,
                    sparse_ids=sparse_tensors.id_tensor,
                    sparse_weights=sparse_tensors.weight_tensor,
                    combiner="sum")
                sigmoid_output_tensor = tf.nn.safe_embedding_lookup_sparse(
                    embedding_weights=sigmoid_weights,
                    sparse_ids=sparse_tensors.id_tensor,
                    sparse_weights=sparse_tensors.weight_tensor,
                    combiner="sum")
            else:
                raise
            softmax_output_tensors.append(softmax_output_tensor)
            sigmoid_output_tensors.append(sigmoid_output_tensor)
        softmax_tensor = tf.nn.bias_add(tf.math.add_n(softmax_output_tensors), self.softmax_bias)
        sigmoid_tensor = tf.nn.bias_add(tf.math.add_n(sigmoid_output_tensors), self.sigmoid_bias)
        softmax_tensor = tf.math.exp(softmax_tensor)
        softmax_tensor = softmax_tensor / tf.math.reduce_sum(softmax_tensor, axis=1, keepdims=True)
        sigmoid_tensor = 1 / (1 + tf.math.exp(sigmoid_tensor))
        logits = tf.math.reduce_sum(tf.math.multiply(softmax_tensor, sigmoid_tensor), axis=1, keepdims=True)
        return logits

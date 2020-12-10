import tensorflow as tf
from tensorflow.python.feature_column.feature_column_lib import FeatureColumn, CategoricalColumn, DenseColumn, \
    CrossedColumn
from tensorflow.python.feature_column.feature_column_v2 import FeatureTransformationCache, _StateManagerImplV2
from typing import List


class Poly2Model(tf.keras.Model):
    def __init__(self, columns: List[FeatureColumn], cross_columns: List[CrossedColumn], l2_factor: float, name=None,
                 **kwargs):
        super(Poly2Model, self).__init__(name=name, **kwargs)
        self.columns = sorted(columns, key=lambda col: col.name)
        self.l2_factor = l2_factor
        self.regularizer = tf.keras.regularizers.l2(l2_factor)
        self.cross_columns = sorted(cross_columns, key=lambda col: col.name)
        self.state_manager = _StateManagerImplV2(self, True)
        self.bias = None

    def build(self, input_shape):
        for column in self.columns:
            column.create_state(self.state_manager)
            if isinstance(column, CategoricalColumn):
                first_dim = column.num_buckets
            elif isinstance(column, DenseColumn):
                first_dim = column.variable_shape.num_elements()
            self.state_manager.create_variable(
                feature_column=column,
                name="weights",
                shape=(first_dim, 1),
                dtype=self.dtype,
                trainable=True)
        for column in self.cross_columns:
            self.state_manager.create_variable(
                feature_column=column,
                name="weights",
                shape=(column.num_buckets, 1),
                dtype=self.dtype,
                trainable=True)
        self.bias = self.add_weight(
            name="bias",
            shape=(1,),
            dtype=self.dtype,
            trainable=True)

    def get_config(self):
        from tensorflow.python.feature_column.feature_column_lib import serialize_feature_columns
        config = {
            "columns": serialize_feature_columns(self.columns),
            "cross_columns": serialize_feature_columns(self.cross_columns)
        }
        base_config = super(Poly2Model, self).get_config()
        return {**base_config, **config}

    def call(self, inputs, training=None, mask=None):
        transformation_cache = FeatureTransformationCache(inputs)
        output_tensors = []
        for column in self.columns:
            column_weight = self.state_manager.get_variable(column, "weights")
            self.add_loss(self.regularizer(column_weight))
            if isinstance(column, CategoricalColumn):
                sparse_tensors = column.get_sparse_tensors(transformation_cache, None)
                output_tensor = tf.nn.safe_embedding_lookup_sparse(
                    embedding_weights=column_weight,
                    sparse_ids=sparse_tensors.id_tensor,
                    sparse_weights=sparse_tensors.weight_tensor,
                    combiner="sum")
                output_tensors.append(output_tensor)
            elif isinstance(column, DenseColumn):
                dense_tensor = column.get_dense_tensor(transformation_cache, self.state_manager)
                output_tensor = tf.matmul(dense_tensor, column_weight)
                output_tensors.append(output_tensor)
        for column in self.cross_columns:
            sparse_tensors = column.get_sparse_tensors(transformation_cache, None)
            column_weight = self.state_manager.get_variable(column, "weights")
            self.add_loss(self.regularizer(column_weight))
            output_tensor = tf.nn.safe_embedding_lookup_sparse(
                embedding_weights=column_weight,
                sparse_ids=sparse_tensors.id_tensor,
                sparse_weights=sparse_tensors.weight_tensor,
                combiner="sum")
            output_tensors.append(output_tensor)
        logits_no_bias = tf.math.add_n(output_tensors)
        logits = tf.nn.bias_add(logits_no_bias, self.bias)
        return logits

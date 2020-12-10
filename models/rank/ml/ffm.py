import tensorflow as tf
from tensorflow.python.feature_column.feature_column_lib import CategoricalColumn
from tensorflow.python.feature_column.feature_column_v2 import FeatureTransformationCache, _StateManagerImplV2
from typing import List


class FFMModel(tf.keras.Model):
    def __init__(self, latent_dim: int, columns: List[CategoricalColumn], name=None, **kwargs):
        super(FFMModel, self).__init__(name=name, **kwargs)
        self.latent_dim = latent_dim
        self.columns = sorted(columns, key=lambda col: col.name)
        self.columns_count = len(self.columns)
        self.state_manager = _StateManagerImplV2(self, trainable=True)
        self.bias = None

    def build(self, input_shape):
        for column in self.columns:
            self.state_manager.create_variable(
                feature_column=column,
                name="weights",
                shape=(column.num_buckets, 1),
                dtype=self.dtype,
                trainable=True)
            self.state_manager.create_variable(
                feature_column=column,
                name="embeddings",
                shape=(self.columns_count - 1, column.num_buckets, self.latent_dim),
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
            "latent_dim": self.latent_dim,
            "columns": serialize_feature_columns(self.columns)
        }
        base_config = super(FFMModel, self).get_config()
        return {**base_config, **config}

    def call(self, inputs, training=None, mask=None):
        transformation_cache = FeatureTransformationCache(inputs)
        linear_tensors = []
        for column in self.columns:
            sparse_tensors = column.get_sparse_tensors(transformation_cache, None)
            linear_tensor = tf.nn.safe_embedding_lookup_sparse(
                embedding_weights=self.state_manager.get_variable(column, "weights"),
                sparse_ids=sparse_tensors.id_tensor,
                sparse_weights=sparse_tensors.weight_tensor,
                combiner="sum")
            linear_tensors.append(linear_tensor)
        cross_tensors = []
        for i in range(self.columns_count - 1):
            column_i = self.columns[i]
            column_i_embeddings = self.state_manager.get_variable(column_i, "embeddings")
            column_i_sparse_tensors = column_i.get_sparse_tensors(transformation_cache, None)
            for j in range(i + 1, self.columns_count):
                column_j = self.columns[j]
                column_j_embeddings = self.state_manager.get_variable(column_j, "embeddings")
                column_j_sparse_tensors = column_j.get_sparse_tensors(transformation_cache, None)
                column_ixj_embedding = tf.nn.safe_embedding_lookup_sparse(
                    embedding_weights=tf.squeeze(tf.slice(column_i_embeddings, begin=[j-1, 0, 0], size=[1, -1, -1]),
                                                 axis=0),
                    sparse_ids=column_i_sparse_tensors.id_tensor,
                    sparse_weights=column_i_sparse_tensors.weight_tensor)
                column_jxi_embedding = tf.nn.safe_embedding_lookup_sparse(
                    embedding_weights=tf.squeeze(tf.slice(column_j_embeddings, begin=[i, 0, 0], size=[1, -1, -1]),
                                                 axis=0),
                    sparse_ids=column_j_sparse_tensors.id_tensor,
                    sparse_weights=column_j_sparse_tensors.weight_tensor)
                cross_tensor = tf.math.multiply(column_ixj_embedding, column_jxi_embedding)
                cross_tensor = tf.math.reduce_sum(cross_tensor, axis=1, keepdims=True)
                cross_tensors.append(cross_tensor)
        logits_no_bias = tf.math.add_n(linear_tensors) + tf.math.add_n(cross_tensors)
        logits = tf.nn.bias_add(logits_no_bias, self.bias)
        return logits

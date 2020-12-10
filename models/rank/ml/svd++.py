import tensorflow as tf
from tensorflow.python.feature_column.feature_column_lib import CategoricalColumn
from tensorflow.python.feature_column.feature_column_v2 import FeatureTransformationCache, _StateManagerImplV2
from typing import List


class SVDPlusPlusModel(tf.keras.Model):
    def __init__(self, average_score: float, latent_dim: int, user_column: CategoricalColumn,
                 item_column: CategoricalColumn, user_history_columns: List[CategoricalColumn],
                 l2_factor_bias: float, l2_factor_embedding: float, name=None, **kwargs):
        super(SVDPlusPlusModel, self).__init__(name=name, **kwargs)
        self.average_score = average_score
        self.latent_dim = latent_dim
        self.user_column = user_column
        self.item_column = item_column
        self.user_history_columns = user_history_columns
        self.l2_factor_bias = l2_factor_bias
        self.l2_factor_embedding = l2_factor_embedding
        self.regularizer_bias = tf.keras.regularizers.l2(l2_factor_bias)
        self.regularizer_embedding = tf.keras.regularizers.l2(l2_factor_embedding)
        self.state_manager = _StateManagerImplV2(self, True)
        self.user_bias = None
        self.item_bias = None

    def get_config(self):
        from tensorflow.python.feature_column.feature_column_lib import serialize_feature_column, \
            serialize_feature_columns
        config = {
            "average_score": self.average_score,
            "latent_dim": self.latent_dim,
            "user_column": serialize_feature_column(self.user_column),
            "item_column": serialize_feature_column(self.item_column),
            "user_history_columns": serialize_feature_columns(self.user_history_columns),
            "l2_factor_bias": self.l2_factor_bias,
            "l2_factor_embedding": self.l2_factor_embedding
        }
        base_config = super(SVDPlusPlusModel, self).get_config()
        return {**base_config, **config}

    def build(self, input_shape):
        self.state_manager.create_variable(
            feature_column=self.user_column,
            name="embeddings",
            shape=(self.user_column.num_buckets, self.latent_dim),
            dtype=self.dtype,
            trainable=True)
        for user_history_column in self.user_history_columns:
            self.state_manager.create_variable(
                feature_column=user_history_column,
                name="embeddings",
                shape=(user_history_column.num_buckets, self.latent_dim),
                dtype=self.dtype,
                trainable=True)
        self.state_manager.create_variable(
            feature_column=self.item_column,
            name="embeddings",
            shape=(self.item_column.num_buckets, self.latent_dim),
            dtype=self.dtype,
            trainable=True)
        self.user_bias = self.add_weight(
            name="user_bias",
            shape=(1,),
            dtype=self.dtype,
            trainable=True)
        self.item_bias = self.add_weight(
            name="item_bias",
            shape=(1,),
            dtype=self.dtype,
            trainable=True)

    def call(self, inputs, training=None, mask=None):
        transformation_cache = FeatureTransformationCache(inputs)
        user_sparse_tensors = self.user_column.get_sparse_tensors(transformation_cache, None)
        user_embedding = tf.nn.safe_embedding_lookup_sparse(
            embedding_weights=self.state_manager.get_variable(self.user_column, "embeddings"),
            sparse_ids=user_sparse_tensors.id_tensor,
            sparse_weights=user_sparse_tensors.weight_tensor)
        self.add_loss(self.regularizer_embedding(user_embedding))
        user_history_embeddings = []
        for user_history_column in self.user_history_columns:
            user_history_sparse_tensors = user_history_column.get_sparse_tensors(transformation_cache, None)
            user_history_embedding = tf.nn.safe_embedding_lookup_sparse(
                embedding_weights=self.state_manager.get_variable(user_history_column, "embeddings"),
                sparse_ids=user_history_sparse_tensors.id_tensor,
                sparse_weights=user_history_sparse_tensors.weight_tensor,
                combiner="sqrtn")
            user_history_embeddings.append(user_history_embedding)
            self.add_loss(self.regularizer_embedding(user_history_embedding))
        user_full_embedding = user_embedding + tf.math.add_n(user_history_embeddings)
        item_sparse_tensors = self.item_column.get_sparse_tensors(transformation_cache, None)
        item_embedding = tf.nn.safe_embedding_lookup_sparse(
            embedding_weights=self.state_manager.get_variable(self.item_column, "embeddings"),
            sparse_ids=item_sparse_tensors.id_tensor,
            sparse_weights=item_sparse_tensors.weight_tensor)
        self.add_loss(self.regularizer_embedding(item_embedding))
        score_no_bias = tf.math.reduce_sum(tf.math.multiply(user_full_embedding, item_embedding), axis=1)
        bias = self.average_score + self.user_bias + self.item_bias
        self.add_loss(self.regularizer_bias(self.user_bias))
        self.add_loss(self.regularizer_bias(self.item_bias))
        score = tf.nn.bias_add(score_no_bias, bias)
        return score

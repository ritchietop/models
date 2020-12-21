import tensorflow as tf
from tensorflow.python.feature_column.feature_column_lib import CategoricalColumn
from tensorflow.python.feature_column.feature_column_v2 import _StateManagerImplV2, FeatureTransformationCache
from typing import List


class EgesModel(tf.keras.Model):
    """
    feature_schema = {
        "item_id": tf.io.FixedLenFeature(shape(1,), dtype=tf.string, default_value=""),
        "item_category": tf.io.FixedLenFeature(shape(1,), dtype=tf.string, default_value=""),
        "item_tags": tf.io.VarLenFeature(dtype=tf.string),
        ...
        "target_id": tf.io.FixedLenFeature(shape(1,), dtype=tf.string, default="")
    }
    """
    def __init__(self,
                 item_embedding_size: int,
                 item_id_column: CategoricalColumn,
                 item_feature_columns: List[CategoricalColumn],
                 target_id_column: CategoricalColumn,
                 name=None, **kwargs):
        super(EgesModel, self).__init__(name=name, **kwargs)

    def call(self, inputs, training=None, mask=None):
        pass

    def get_config(self):
        pass


class ItemWeightedFeatureLayer(tf.keras.layers.Layer):
    def __init__(self,
                 item_embedding_size: int,
                 item_id_column: CategoricalColumn,
                 item_feature_columns: List[CategoricalColumn],
                 trainable=True, name=None, **kwargs):
        super(ItemWeightedFeatureLayer, self).__init__(trainable=trainable, name=name, **kwargs)
        self.item_embedding_size = item_embedding_size
        self.item_id_column = item_id_column
        self.item_feature_columns = item_feature_columns
        self.state_manager = _StateManagerImplV2(self, self.trainable)
        self.feature_weights = None

    def build(self, input_shape):
        with tf.name_scope(self.item_id_column.name):
            self.state_manager.create_variable(
                self.item_id_column,
                name="item_id_embedding",
                shape=(self.item_id_column.num_buckets, self.item_embedding_size),
                dtype=tf.float32,
                trainable=self.trainable)
        for item_feature_column in self.item_feature_columns:
            self.state_manager.create_variable(
                item_feature_column,
                name="%s_embedding" % item_feature_column.name,
                shape=(item_feature_column.num_buckets, self.item_embedding_size),
                dtype=tf.float32,
                trainable=self.trainable)
        self.feature_weights = self.add_weight(
            name="item_feature_weights",
            shape=(self.item_id_column.num_buckets, len(self.item_feature_columns) + 1),
            dtype=tf.float32,
            trainable=self.trainable)

    def call(self, inputs, **kwargs):
        transformation_cache = FeatureTransformationCache(inputs)
        item_feature_tensors = []
        item_id_sparse_tensors = self.item_id_column.get_sparse_tensors(transformation_cache, None)
        item_id_tensor = tf.nn.safe_embedding_lookup_sparse(
            embedding_weights=self.state_manager.get_variable(self.item_id_column, "item_id_embedding"),
            sparse_ids=item_id_sparse_tensors.id_tensor,
            sparse_weights=item_id_sparse_tensors.weight_tensor)
        item_feature_tensors.append(item_id_tensor)
        for item_feature_column in self.item_feature_columns:
            item_feature_sparse_tensors = item_feature_column.get_sparse_tensors(transformation_cache, None)
            item_feature_tensor = tf.nn.safe_embedding_lookup_sparse(
                embedding_weights=self.state_manager.get_variable(item_feature_column,
                                                                  "%s_embedding" % item_feature_column.name),
                sparse_ids=item_feature_sparse_tensors.id_tensor,
                sparse_weights=item_feature_sparse_tensors.weight_tensor,
                combiner="mean")
            item_feature_tensors.append(item_feature_tensor)
        item_features_tensor = tf.stack(item_feature_tensors, axis=1)
        with tf.name_scope("WeightedItemFeature"):
            item_weights = tf.nn.safe_embedding_lookup_sparse(
                embedding_weights=self.feature_weights,
                sparse_ids=item_id_sparse_tensors.id_tensor)
            item_weights_exp = tf.math.exp(item_weights)
            item_weights = item_weights_exp / tf.reduce_sum(item_weights_exp)
        weighted_item_features_tensor = tf.matmul(item_features_tensor)

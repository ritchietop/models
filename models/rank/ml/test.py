import tensorflow as tf
from tensorflow.python.feature_column.feature_column_lib import CategoricalColumn
from tensorflow.python.feature_column.feature_column_v2 import FeatureTransformationCache, _StateManagerImplV2
from absl import app


class SVDModel(tf.keras.Model):
    def __init__(self, latent_dim: int, user_column: CategoricalColumn,
                 item_column: CategoricalColumn, l2_factor: float, name=None, **kwargs):
        super(SVDModel, self).__init__(name=name, **kwargs)
        self.latent_dim = latent_dim
        self.user_column = user_column
        self.item_column = item_column
        self.l2_factor = l2_factor
        self.regularizer = tf.keras.regularizers.l2(l2_factor)
        self.state_manager = _StateManagerImplV2(self, self.trainable)
        self.average_score = None
        self.user_bias = None
        self.item_bias = None

    def get_config(self):
        from tensorflow.python.feature_column.feature_column_lib import serialize_feature_column
        config = {
            "average_score": self.average_score,
            "latent_dim": self.latent_dim,
            "user_column": serialize_feature_column(self.user_column),
            "item_column": serialize_feature_column(self.item_column),
            "l2_factor": self.l2_factor
        }
        base_config = super(SVDModel, self).get_config()
        return {**base_config, **config}

    def build(self, input_shape):
        self.state_manager.create_variable(
            feature_column=self.user_column,
            name="embeddings",
            shape=(self.user_column.num_buckets, self.latent_dim),
            initializer=tf.keras.initializers.random_uniform,
            dtype=self.dtype,
            trainable=True)
        self.state_manager.create_variable(
            feature_column=self.item_column,
            name="embeddings",
            shape=(self.item_column.num_buckets, self.latent_dim),
            initializer=tf.keras.initializers.random_uniform,
            dtype=self.dtype,
            trainable=True)
        self.average_score = self.add_weight(name="average_score", shape=(1,), dtype=self.dtype, trainable=True)
        self.user_bias = self.add_weight(
            name="user_bias",
            shape=(self.user_column.num_buckets, 1),
            dtype=self.dtype,
            trainable=True)
        self.item_bias = self.add_weight(
            name="item_bias",
            shape=(self.item_column.num_buckets, 1),
            dtype=self.dtype,
            trainable=True)

    def call(self, inputs, training=None, mask=None):
        transformation_cache = FeatureTransformationCache(inputs)
        user_sparse_tensors = self.user_column.get_sparse_tensors(transformation_cache, None)
        user_embedding = tf.nn.safe_embedding_lookup_sparse(
            embedding_weights=self.state_manager.get_variable(self.user_column, "embeddings"),
            sparse_ids=user_sparse_tensors.id_tensor,
            sparse_weights=user_sparse_tensors.weight_tensor)
        self.add_loss(self.regularizer(user_embedding))
        item_sparse_tensors = self.item_column.get_sparse_tensors(transformation_cache, None)
        item_embedding = tf.nn.safe_embedding_lookup_sparse(
            embedding_weights=self.state_manager.get_variable(self.item_column, "embeddings"),
            sparse_ids=item_sparse_tensors.id_tensor,
            sparse_weights=item_sparse_tensors.weight_tensor)
        self.add_loss(self.regularizer(item_embedding))
        score_no_bias = tf.math.reduce_sum(tf.math.multiply(user_embedding, item_embedding), axis=1, keepdims=True)
        user_bias = tf.nn.safe_embedding_lookup_sparse(
            embedding_weights=self.user_bias,
            sparse_ids=user_sparse_tensors.id_tensor)
        item_bias = tf.nn.safe_embedding_lookup_sparse(
            embedding_weights=self.item_bias,
            sparse_ids=item_sparse_tensors.id_tensor)
        self.add_loss(self.regularizer(self.user_bias))
        self.add_loss(self.regularizer(self.item_bias))
        score = tf.nn.bias_add(score_no_bias + user_bias + item_bias, self.average_score)
        return score_no_bias + user_bias + item_bias


def main(_):
    import os
    movie_ratings_path = os.path.abspath(__file__).replace("models/rank/ml/test.py", "data/movieLens/ml-100k/u.data")
    data = tf.data.experimental.make_csv_dataset(
        file_pattern=movie_ratings_path,
        batch_size=64,
        column_names=["user_id",  "item_id",  "rating",  "timestamp"],
        select_columns=["user_id", "item_id", "rating"],
        column_defaults=[0, 0, 0.0],
        label_name="rating",
        field_delim="\t",
        use_quote_delim=True,
        na_value="null",
        header=False,
        num_epochs=1,
        shuffle=bool(100),
        shuffle_buffer_size=100,
        num_rows_for_inference=0,
        ignore_errors=False)

    model = SVDModel(latent_dim=5,
                     user_column=tf.feature_column.categorical_column_with_identity(key="user_id", num_buckets=10000),
                     item_column=tf.feature_column.categorical_column_with_identity(key="item_id", num_buckets=10000),
                     l2_factor=0)
    model.compile(optimizer='Adam',
                  loss=tf.keras.losses.mean_squared_error,
                  metrics=tf.keras.metrics.RootMeanSquaredError())
    model.fit(x=data, epochs=64, callbacks=[tf.keras.callbacks.TensorBoard(log_dir="./svd")])


if __name__ == "__main__":
    app.run(main)

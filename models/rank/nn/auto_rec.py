import tensorflow as tf
from tensorflow.python.feature_column.feature_column_lib import WeightedCategoricalColumn, FeatureTransformationCache


class AutoRecModel(tf.keras.Model):
    '''
        h(r;x) = f(W * g(Vr + u) + b)
        loss = min{sum( ||r(i) - h(r(i);x)||^2 + t / 2 * (||W||^2 + ||V||^2)
    '''
    def __init__(self, rating_column: WeightedCategoricalColumn, latent_dim: int, encoder_activation,
                 decoder_activation, l2_factor: float, name=None, **kwargs):
        super(AutoRecModel, self).__init__(name=name, **kwargs)
        self.rating_column = rating_column
        self.latent_dim = latent_dim
        self.encoder_activation = encoder_activation
        self.decoder_activation = decoder_activation
        self.l2_factor = l2_factor
        self.encoder_weights = None
        self.decoder_weights = None
        self.encoder_bias = None
        self.decoder_bias = None

    def get_config(self):
        from tensorflow.python.feature_column.feature_column_lib import serialize_feature_column
        config = {
            "rating_column": serialize_feature_column(self.rating_column),
            "latent_dim": self.latent_dim,
            "encoder_activation": tf.keras.utils.serialize_keras_object(self.encoder_activation),
            "decoder_activation": tf.keras.utils.serialize_keras_object(self.decoder_activation),
            "l2_factor": self.l2_factor
        }
        base_config = super(AutoRecModel, self).get_config()
        return {**base_config, **config}

    def build(self, input_shape):
        self.encoder_weights = self.add_weight(
            name="encoder_weights",
            shape=(self.rating_column.num_buckets, self.latent_dim),
            dtype=self.dtype,
            trainable=True,
            regularizer=tf.keras.regularizers.l2(self.l2_factor))
        self.decoder_weights = self.add_weight(
            name="decoder_weights",
            shape=(self.latent_dim, self.rating_column.num_buckets),
            dtype=self.dtype,
            trainable=True,
            regularizer=tf.keras.regularizers.l2(self.l2_factor))
        self.encoder_bias = self.add_weight(
            name="encoder_bias",
            shape=(self.latent_dim,),
            dtype=self.dtype,
            trainable=True)
        self.decoder_bias = self.add_weight(
            name="decoder_bias",
            shape=(self.rating_column.num_buckets,),
            dtype=self.dtype,
            trainable=True)

    def call(self, inputs, training=None, mask=None):
        transformation_cache = FeatureTransformationCache(data)
        sparse_tensors = self.rating_column.get_sparse_tensors(transformation_cache, None)
        encoder_tensor = tf.nn.safe_embedding_lookup_sparse(
            embedding_weights=self.encoder_weights,
            sparse_ids=sparse_tensors.id_tensor,
            sparse_weights=sparse_tensors.weight_tensor,
            combiner="sum")
        encoder_tensor = tf.nn.bias_add(encoder_tensor, self.encoder_bias)
        encoder_tensor = self.encoder_activation(encoder_tensor)
        decoder_tensor = tf.matmul(encoder_tensor, self.decoder_weights)
        decoder_tensor = tf.nn.bias_add(decoder_tensor, self.decoder_bias)
        decoder_tensor = self.decoder_activation(decoder_tensor)
        return decoder_tensor, sparse_tensors

    def train_step(self, data):
        with tf.GradientTape() as tape:
            decoder_tensor, sparse_tensors = self(data, training=True)
            input_rating = sparse_tensors.weight_tensor
            output_rating = tf.gather_nd(params=decoder_tensor, indices=sparse_tensors.id_tensor, batch_dims=0)
            loss = self.compiled_loss(input_rating, output_rating, regularization_losses=self.losses)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {m.name: m.result() for m in self.metrics}

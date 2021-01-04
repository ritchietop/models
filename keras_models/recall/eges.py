import tensorflow as tf
from absl import app
from data.movieLens_graph import input_fn


def eges_model(embedding_size, num_sampled=10, item_id_size=3953):
    target_movie_id = tf.keras.layers.Input(shape=(1,), name="target_movie_id", dtype=tf.int64)
    movie_id = tf.keras.layers.Input(shape=(1,), name="movie_id", dtype=tf.int64)
    keywords = tf.keras.layers.Input(shape=(None,), name="keywords", dtype=tf.string, ragged=True)
    publish_year = tf.keras.layers.Input(shape=(1,), name="publishYear", dtype=tf.int64)
    categories = tf.keras.layers.Input(shape=(None,), name="categories", dtype=tf.string, ragged=True)

    movie_id_layer = tf.keras.layers.Embedding(input_dim=item_id_size, output_dim=embedding_size)(movie_id)
    keywords_layer = tf.keras.layers.Lambda(function=lambda tensor: tf.expand_dims(embedding_pooling(tensor, "sqrtn"),
                                                                                   axis=1),
                                            name="KeywordsPoolingLayer")(
        tf.keras.layers.Embedding(input_dim=10000, output_dim=embedding_size)(
            tf.keras.layers.experimental.preprocessing.Hashing(num_bins=10000)(keywords)))
    publish_year_layer = tf.keras.layers.Embedding(input_dim=82, output_dim=embedding_size)(
        tf.keras.layers.experimental.preprocessing.IntegerLookup(
            vocabulary=list(range(1919, 2001)), mask_value=None, num_oov_indices=0)(publish_year))
    categories_layer = tf.keras.layers.Lambda(function=lambda tensor: tf.expand_dims(embedding_pooling(tensor, "sqrtn"),
                                                                                     axis=1),
                                              name="CategoryPoolingLayer")(
        tf.keras.layers.Embedding(input_dim=18, output_dim=embedding_size)(
            tf.keras.layers.experimental.preprocessing.StringLookup(
                vocabulary=["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama",
                            "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller",
                            "War", "Western"], mask_token=None, num_oov_indices=0)(categories)))

    movie_embedding_layer_inputs = [movie_id_layer, keywords_layer, publish_year_layer, categories_layer]

    movie_embedding_layer = tf.keras.layers.Concatenate(axis=1)(inputs=movie_embedding_layer_inputs)

    movie_embedding_weight_layer = tf.keras.layers.Embedding(
        input_dim=item_id_size, output_dim=len(movie_embedding_layer_inputs), name="FeatureWeightLayer")(movie_id)
    movie_embedding_normalize_weight_layer = tf.keras.layers.Lambda(
        function=normalize_embedding_weight, name="NormalizeFeatureWeightLayer")(movie_embedding_weight_layer)
    weighted_movie_embedding_layer = tf.keras.layers.Lambda(
        function=weighted_movie_embedding, name="WeightedMovieEmbeddingLayer")(
            inputs=[movie_embedding_normalize_weight_layer, movie_embedding_layer])

    loss = CandidateSampledLossLayer(item_id_size=item_id_size, embedding_size=embedding_size, num_sampled=num_sampled)(
        inputs=[weighted_movie_embedding_layer, target_movie_id])

    model = tf.keras.Model(inputs=[target_movie_id, movie_id, keywords, publish_year, categories], outputs=loss)

    return model


def normalize_embedding_weight(tensor):
    tensor = tf.math.exp(tensor)
    tensor_sum = tf.math.reduce_sum(tensor, axis=1, keepdims=True)
    return tf.transpose(tensor / tensor_sum, perm=[0, 2, 1])


def weighted_movie_embedding(inputs):
    embedding, weight = inputs
    return tf.math.reduce_sum(embedding * weight, axis=1)


def embedding_pooling(tensor: tf.RaggedTensor, combiner: str = "sqrtn"):
    tensor_sum = tf.math.reduce_sum(tensor, axis=1)
    if combiner == "sum":
        return tensor_sum
    row_lengths = tf.expand_dims(tensor.row_lengths(axis=1), axis=1)
    row_lengths = tf.math.maximum(tf.ones_like(row_lengths), row_lengths)
    row_lengths = tf.cast(row_lengths, dtype=tf.float32)
    if combiner == "mean":
        return tensor_sum / row_lengths
    if combiner == "sqrtn":
        return tensor_sum / tf.math.sqrt(row_lengths)


class CandidateSampledLossLayer(tf.keras.layers.Layer):
    def __init__(self,
                 item_id_size: int,
                 embedding_size: int,
                 num_sampled: int,
                 num_true: int = 1,
                 trainable=True, name=None, **kwargs):
        super(CandidateSampledLossLayer, self).__init__(trainable=trainable, name=name, **kwargs)
        self.item_id_size = item_id_size
        self.target_movie_ids_embeddings = self.add_weight(name="embedding", shape=(item_id_size, embedding_size))
        self.target_movie_ids_bias = self.add_weight(name="bias_embedding", shape=(item_id_size,),
                                                     initializer=tf.keras.initializers.zeros)
        self.num_sampled = num_sampled
        self.num_true = num_true

    def call(self, inputs, training=None):
        movie_id_tensor, target_movie_ids = inputs
        if training:
            loss = tf.nn.sampled_softmax_loss(
                weights=self.target_movie_ids_embeddings,
                biases=self.target_movie_ids_bias,
                labels=target_movie_ids,
                inputs=movie_id_tensor,
                num_sampled=self.num_sampled,
                num_classes=self.item_id_size,
                num_true=self.num_true)
        else:
            logits = tf.matmul(movie_id_tensor, self.target_movie_ids_embeddings, transpose_b=True)
            logits = tf.nn.bias_add(logits, self.target_movie_ids_bias)
            labels = tf.one_hot(indices=target_movie_ids, depth=self.item_id_size)
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        return loss


def main(_):
    model = eges_model(embedding_size=128, num_sampled=10)
    train_data = input_fn(batch_size=500, shuffle_buffer_size=1000)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.mean_squared_error,
                  metrics=tf.keras.metrics.RootMeanSquaredError())
    model.fit(x=train_data, epochs=1)
    tf.keras.utils.plot_model(model, to_file="./eges.png", rankdir="BT")


if __name__ == "__main__":
    app.run(main)

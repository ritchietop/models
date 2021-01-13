import tensorflow as tf
from absl import app
from data.movieLens import train_input_fn, test_input_fn, user_occupation
from typing import List


def din_model(hidden_units: List[int], dropout: float):
    # context inputs
    timestamp = tf.keras.layers.Input(shape=(1,), name="timestamp", dtype=tf.int64)
    # user inputs
    gender = tf.keras.layers.Input(shape=(1,), name="gender", dtype=tf.string)
    age = tf.keras.layers.Input(shape=(1,), name="age", dtype=tf.int64)
    occupation = tf.keras.layers.Input(shape=(1,), name="occupation", dtype=tf.string)
    zip_code = tf.keras.layers.Input(shape=(1,), name="zip_code", dtype=tf.string)
    # item inputs
    movie_id = tf.keras.layers.Input(shape=(1,), name="movie_id", dtype=tf.int64)
    keywords = tf.keras.layers.Input(shape=(None,), name="keywords", dtype=tf.string, ragged=True)
    categories = tf.keras.layers.Input(shape=(None,), name="categories", dtype=tf.string, ragged=True)
    # behavior inputs
    user_history_high_score_movies = tf.keras.layers.Input(
        shape=(None,), name="user_history_high_score_movies", dtype=tf.int64, ragged=True)
    user_history_high_score_movie_keywords = tf.keras.layers.Input(
        shape=(None,), name="user_history_high_score_movie_keywords", dtype=tf.string, ragged=True)
    user_history_high_score_movie_categories = tf.keras.layers.Input(
        shape=(None,), name="user_history_high_score_movie_categories", dtype=tf.string, ragged=True)

    # context features
    timestamp_hour_layer = tf.keras.layers.Embedding(input_dim=24, output_dim=5)(
        tf.keras.layers.experimental.preprocessing.Discretization(bins=list(range(24)))(
            tf.keras.layers.Lambda(function=lambda tensor: tensor // 3600 % 24)(timestamp)))
    timestamp_week_layer = tf.keras.layers.Embedding(input_dim=7, output_dim=3)(
        tf.keras.layers.experimental.preprocessing.Discretization(bins=list(range(7)))(
            tf.keras.layers.Lambda(function=lambda tensor: tensor // 86400 % 7)(timestamp)))
    context_profile_embed_layer = tf.keras.layers.Concatenate()(inputs=[timestamp_hour_layer, timestamp_week_layer])

    # user profile features
    gender_embed_layer = tf.keras.layers.Embedding(input_dim=2, output_dim=5)(
        tf.keras.layers.experimental.preprocessing.StringLookup(
            vocabulary=["F", "M"], mask_token=None, num_oov_indices=0)(gender))
    age_embed_layer = tf.keras.layers.Embedding(input_dim=7, output_dim=5)(
        tf.keras.layers.experimental.preprocessing.IntegerLookup(
            vocabulary=[1, 18, 25, 35, 45, 50, 56], mask_value=None, num_oov_indices=0)(age))
    occupation_embed_layer = tf.keras.layers.Embedding(input_dim=21, output_dim=10)(
        tf.keras.layers.experimental.preprocessing.StringLookup(
            vocabulary=list(user_occupation.values()), mask_token=None, num_oov_indices=0)(occupation))
    # 3439
    zip_code_embed_layer = tf.keras.layers.Embedding(input_dim=10000, output_dim=15)(
        tf.keras.layers.experimental.preprocessing.Hashing(num_bins=10000)(zip_code))
    user_profile_embed_layer = tf.keras.layers.Concatenate()(inputs=[
        gender_embed_layer, age_embed_layer, occupation_embed_layer, zip_code_embed_layer
    ])

    # common layers of behavior and candidate
    # # id
    movie_id_embedding_size = 128
    id_embed_layer = tf.keras.layers.Embedding(input_dim=3953, output_dim=movie_id_embedding_size)
    # # keyword
    movie_keyword_embedding_size = 128
    keywords_layer = tf.keras.layers.experimental.preprocessing.Hashing(num_bins=10000)
    keyword_embed_layer = tf.keras.layers.Embedding(input_dim=10000, output_dim=movie_keyword_embedding_size)
    # # category
    category_embedding_size = 5
    categories_layer = tf.keras.layers.experimental.preprocessing.StringLookup(
        vocabulary=["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama",
                    "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War",
                    "Western"], mask_token=None, num_oov_indices=0)
    categories_embed_layer = tf.keras.layers.Embedding(input_dim=18, output_dim=category_embedding_size)
    # # candidate pooling
    embedding_pooling_layer = tf.keras.layers.Lambda(lambda tensor: embedding_pooling(tensor))
    # # behavior pooling
    multi_behavior_pooling_layer = tf.keras.layers.Lambda(lambda tensor: multi_behavior_embedding_pooling(tensor))

    # candidate item profile features
    candidate_id_embed_layer = id_embed_layer(movie_id)
    candidate_id_embed_layer = embedding_pooling_layer(candidate_id_embed_layer)
    candidate_keywords_embed_layer = keyword_embed_layer(keywords_layer(keywords))
    candidate_keywords_embed_layer = embedding_pooling_layer(candidate_keywords_embed_layer)
    candidate_categories_embed_layer = categories_embed_layer(categories_layer(categories))
    candidate_categories_embed_layer = embedding_pooling_layer(candidate_categories_embed_layer)
    # batch_size * 1 * embedding_size
    candidate_profile_embed_layer = tf.keras.layers.Concatenate(axis=-1)(inputs=[
        candidate_id_embed_layer, candidate_keywords_embed_layer, candidate_categories_embed_layer
    ])

    # behavior features
    behavior_id_embed_layer = id_embed_layer(user_history_high_score_movies)
    behavior_id_embed_layer = multi_behavior_pooling_layer(behavior_id_embed_layer)
    behavior_keywords_embed_layer = keyword_embed_layer(keywords_layer(user_history_high_score_movie_keywords))
    behavior_keywords_embed_layer = multi_behavior_pooling_layer(behavior_keywords_embed_layer)
    behavior_categories_embed_layer = categories_embed_layer(categories_layer(user_history_high_score_movie_categories))
    behavior_categories_embed_layer = multi_behavior_pooling_layer(behavior_categories_embed_layer)
    # batch_size * behavior_count * embedding_size
    behavior_profile_embed_layer = tf.keras.layers.Concatenate(axis=-1)(inputs=[
        behavior_id_embed_layer, behavior_keywords_embed_layer, behavior_categories_embed_layer
    ])

    # attention be

    model = tf.keras.Model(inputs=[
        timestamp, gender, age, occupation, zip_code, movie_id, keywords, categories,
        user_history_high_score_movies, user_history_high_score_movie_keywords,
        user_history_high_score_movie_categories
    ], outputs=[
        # context_profile_embed_layer, user_profile_embed_layer, candidate_profile_embed_layer,
        # behavior_profile_embed_layer
        candidate_profile_embed_layer
    ])

    return model


# tensor: [batch_size, record_count, embedding_size]
def embedding_pooling(tensor: tf.RaggedTensor, combiner: str = "sqrtn"):
    tensor_sum = tf.math.reduce_sum(tensor, axis=1)  # batch_size * embedding_size
    if combiner == "sum":
        return tensor_sum
    if isinstance(tensor, tf.RaggedTensor):
        row_lengths = tf.expand_dims(tensor.row_lengths(axis=1), axis=1)  # batch_size * 1
        row_lengths = tf.math.maximum(tf.ones_like(row_lengths), row_lengths)
    elif isinstance(tensor, tf.Tensor):
        row_lengths = tf.squeeze(tf.ones_like(tensor), axis=2) * tensor.shape[1]
        print(row_lengths)
    else:
        raise ValueError("Only support Tensor or RaggedTensor.")
    row_lengths = tf.cast(row_lengths, dtype=tf.float32)
    if combiner == "mean":
        return tensor_sum / row_lengths
    if combiner == "sqrtn":
        return tensor_sum / tf.math.sqrt(row_lengths)


# tensor1: [batch_size, behavior_count, embedding_size]
# tensor2: [batch_size, behavior_count, record_count, embedding_size]
# return shape: [batch_size, behavior_count, embedding_size]
def multi_behavior_embedding_pooling(tensor: tf.RaggedTensor, combiner: str = "sqrtn"):
    if len(tensor.shape) == 3:
        tensor = tf.expand_dims(tensor, axis=2)  # batch_size * behavior_count * 1 * embedding_size
    tensor_sum = tf.math.reduce_sum(tensor, axis=2)  # batch_size * behavior_count * embedding_size
    if combiner == "sum":
        return tensor_sum
    row_lengths = tf.expand_dims(tensor.row_lengths(axis=2), axis=2)
    row_lengths = tf.math.maximum(tf.ones_like(row_lengths), row_lengths)
    row_lengths = tf.cast(row_lengths, dtype=tf.float32)
    if combiner == "mean":
        return tensor_sum / row_lengths
    if combiner == "sqrtn":
        return tensor_sum / tf.math.sqrt(row_lengths)


class AttentionUnitLayer(tf.keras.layers.Layer):
    def __init__(self, trainable=True, name=None, **kwargs):
        super(AttentionUnitLayer, self).__init__(trainable=trainable, name=name, **kwargs)


class DiceActivation(tf.keras.layers.Layer):
    def __init__(self, axis=-1, epsilon=0.0000000001, trainable=True, name=None, **kwargs):
        super(DiceActivation, self).__init__(trainable=trainable, name=name, **kwargs)
        self.axis = axis
        self.epsilon = epsilon
        self.alpha = None

    def build(self, input_shape):
        self.alpha = self.add_weight(name="alpha",
                                     shape=(input_shape[-1]),
                                     initializer=tf.keras.initializers.constant(0.0),
                                     dtype=tf.float32)

    def call(self, inputs, **kwargs):
        mean = tf.math.reduce_mean(inputs, axis=self.axis, keepdims=True)
        std = tf.math.sqrt(tf.math.reduce_mean(tf.math.square(inputs - mean) + self.epsilon, axis=self.axis,
                                               keepdims=True))
        p = tf.nn.sigmoid((inputs - mean) / (std + self.epsilon))
        return self.alpha * (1.0 - p) * inputs + p * inputs

    def get_config(self):
        config = {
            "axis": self.axis,
            "epsilon": self.epsilon
        }
        base_config = super(DiceActivation, self).get_config()
        return {**base_config, **config}


def main(_):
    model = din_model(hidden_units=[128, 128, 128], dropout=0.5)
    train_data = train_input_fn(batch_size=2)
    # validate_data = test_input_fn(batch_size=1000)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.mean_squared_error,
                  metrics=tf.keras.metrics.RootMeanSquaredError())
    # model.fit(x=train_data, validation_data=validate_data, epochs=1)
    # tf.keras.utils.plot_model(model, to_file="./din.png", rankdir="BT")
    for features, _ in train_data:
        xx = model(features)
        #print(features["user_history_high_score_movie_keywords"])
        #print(xx)
        print([x.shape for x in xx])
        # print(features["user_history_low_score_movie_keywords"].shape)
        # print(behavior_keywords_embed_layer.shape)
        # tensor_sum = tf.math.reduce_sum(behavior_keywords_embed_layer, axis=-2)
        # print(tensor_sum.shape)
        # print(behavior_keywords_embed_layer.row_lengths(axis=-2))
        # row_lengths = tf.expand_dims(behavior_keywords_embed_layer.row_lengths(axis=-2), axis=-1)
        # print(row_lengths.shape)
        # print(tf.math.maximum(tf.ones_like(row_lengths), row_lengths))
        # combiner = "sum"
        # if combiner == "sum":
        #     return tensor_sum
        # row_lengths = tf.expand_dims(tensor.row_lengths(axis=1), axis=1)
        # row_lengths = tf.math.maximum(tf.ones_like(row_lengths), row_lengths)
        # row_lengths = tf.cast(row_lengths, dtype=tf.float32)
        # if combiner == "mean":
        #     return tensor_sum / row_lengths
        # if combiner == "sqrtn":
        #     return tensor_sum / tf.math.sqrt(row_lengths)
        break


if __name__ == "__main__":
    app.run(main)

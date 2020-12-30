import tensorflow as tf
from absl import app
from data.movieLens import train_input_fn, test_input_fn, user_occupation

from typing import List


def youtube_net_model(embedding_size,
                      hidden_units: List[int],
                      dropout: float,
                      activation_fn=tf.keras.activations.relu,
                      num_sampled: int = 100,
                      movie_id_size=3953):
    assert embedding_size == hidden_units[-1], "隐含层最后一层需要和嵌入向量维度保持一致"

    # movie
    movie_id = tf.keras.layers.Input(shape=(1,), name="movie_id", dtype=tf.int64)
    # user
    gender = tf.keras.layers.Input(shape=(1,), name="gender", dtype=tf.string)
    age = tf.keras.layers.Input(shape=(1,), name="age", dtype=tf.int64)
    occupation = tf.keras.layers.Input(shape=(1,), name="occupation", dtype=tf.string)
    zip_code = tf.keras.layers.Input(shape=(1,), name="zip_code", dtype=tf.string)
    user_history_high_score_movies = tf.keras.layers.Input(
        shape=(None,), name="user_history_high_score_movies", dtype=tf.int64, ragged=True)
    user_history_low_score_movies = tf.keras.layers.Input(
        shape=(None,), name="user_history_low_score_movies", dtype=tf.int64, ragged=True)

    movie_id_embedding_layer = tf.keras.layers.Embedding(input_dim=movie_id_size, output_dim=embedding_size)

    # user input features
    gender_layer = tf.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens=2, output_mode="binary")(
        tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=["F", "M"], num_oov_indices=0,
                                                                mask_token=None)(gender))
    age_layer = tf.keras.layers.Flatten()(
        tf.keras.layers.Embedding(input_dim=8, output_dim=3)(
            tf.keras.layers.experimental.preprocessing.IntegerLookup(
                vocabulary=[1, 18, 25, 35, 45, 50, 56], mask_value=None)(age)))
    occupation_layer = tf.keras.layers.Flatten()(
        tf.keras.layers.Embedding(input_dim=22, output_dim=5)(
            tf.keras.layers.experimental.preprocessing.StringLookup(
                vocabulary=list(user_occupation.values()), mask_token=None)(occupation)))
    zip_code_layer = tf.keras.layers.Flatten()(
            tf.keras.layers.Embedding(input_dim=10000, output_dim=13)(
            tf.keras.layers.experimental.preprocessing.Hashing(num_bins=10000)(zip_code)))
    user_history_high_score_movies_layer = tf.keras.layers.Lambda(
        function=lambda tensor: tf.math.reduce_mean(tensor, axis=1))(
        movie_id_embedding_layer(user_history_high_score_movies))
    user_history_low_score_movies_layer = tf.keras.layers.Lambda(
        function=lambda tensor: tf.math.reduce_mean(tensor, axis=1))(
        movie_id_embedding_layer(user_history_low_score_movies))

    user_layer = tf.keras.layers.Concatenate()(inputs=[
        gender_layer, age_layer, occupation_layer, zip_code_layer, user_history_high_score_movies_layer,
        user_history_low_score_movies_layer
    ])

    for hidden_unit in hidden_units:
        user_layer = tf.keras.layers.Dense(units=hidden_unit, activation=activation_fn)(user_layer)
        user_layer = tf.keras.layers.Dropout(rate=dropout)(user_layer)

    loss = CandidateSampledLossLayer(movie_id_embedding_layer, num_sampled)(inputs=[user_layer, movie_id])

    model = tf.keras.models.Model(inputs=[
        movie_id, gender, age, occupation, zip_code, user_history_high_score_movies, user_history_low_score_movies
    ], outputs=[loss, gender_layer, age_layer, occupation_layer, zip_code_layer, user_history_high_score_movies_layer,
        user_history_low_score_movies_layer])

    return model


class CandidateSampledLossLayer(tf.keras.layers.Layer):
    def __init__(self,
                 movie_id_layer: tf.keras.layers.Embedding,
                 num_sampled: int,
                 num_true: int = 1,
                 trainable=True, name=None, **kwargs):
        super(CandidateSampledLossLayer, self).__init__(trainable=trainable, name=name, **kwargs)
        self.movie_id_layer = movie_id_layer
        self.movie_id_bias = self.add_weight(name="bias_embedding", shape=(movie_id_layer.input_dim,),
                                             initializer=tf.keras.initializers.zeros)
        self.num_sampled = num_sampled
        self.num_true = num_true

    @property
    def movie_id_embeddings(self):
        return self.movie_id_layer.trainable_weights[0]

    def call(self, inputs, training=None):
        user_tensor, movie_id = inputs
        if training:
            loss = tf.nn.sampled_softmax_loss(
                weights=self.movie_id_embeddings,
                biases=self.movie_id_bias,
                labels=movie_id,
                inputs=user_tensor,
                num_sampled=self.num_sampled,
                num_classes=self.movie_id_layer.input_dim,
                num_true=self.num_true)
        else:
            logits = tf.matmul(user_tensor, self.movie_id_embeddings, transpose_b=True)
            logits = tf.nn.bias_add(logits, self.movie_id_bias)
            labels = tf.one_hot(indices=movie_id, depth=self.movie_id_layer.input_dim)
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        return loss


def main(_):
    model = youtube_net_model(embedding_size=128, hidden_units=[256, 256, 128], dropout=0.5)
    train_data = train_input_fn(batch_size=10, label_key="label")
    validate_data = test_input_fn(batch_size=1000, label_key="label")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.mean_squared_error,
                  metrics=tf.keras.metrics.AUC(), run_eagerly=True)
    # model.fit(x=train_data, validation_data=validate_data, epochs=2)
    # tf.keras.utils.plot_model(model, to_file="./youtube_net.png", rankdir="BT")
    for data, label in train_data:
        print(data)
        print(model(data))
        print(label)
        break


if __name__ == "__main__":
    app.run(main)

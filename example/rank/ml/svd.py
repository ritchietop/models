import tensorflow as tf
from absl import app
from data.movieLens import train_input_fn, test_input_fn


def svd_model(average_score, embedding_size, l2_factor, user_id_size=6041, item_id_size=3953):
    user_id = tf.keras.layers.Input(shape=(1,), name="user_id", dtype=tf.int64)
    movie_id = tf.keras.layers.Input(shape=(1,), name="movie_id", dtype=tf.int64)

    user_id_embedding_input = tf.keras.layers.Embedding(
        input_dim=user_id_size, output_dim=embedding_size,
        embeddings_regularizer=tf.keras.regularizers.l2(l2_factor))(user_id)
    movie_id_embedding_input = tf.keras.layers.Embedding(
        input_dim=item_id_size, output_dim=embedding_size,
        embeddings_regularizer=tf.keras.regularizers.l2(l2_factor))(movie_id)

    predict_score = tf.keras.layers.Dot(axes=-1)(inputs=[user_id_embedding_input, movie_id_embedding_input])
    predict_bias = AddBiasLayer(global_average_score=average_score, user_id_size=user_id_size,
                                item_id_size=item_id_size, l2_factor=l2_factor)([user_id, movie_id])
    predict = tf.keras.layers.Add()(inputs=[predict_score, predict_bias])

    model = tf.keras.Model(inputs=[user_id, movie_id], outputs=[predict])
    return model


class AddBiasLayer(tf.keras.layers.Layer):
    def __init__(self,
                 global_average_score: float,
                 user_id_size: int,
                 item_id_size: int,
                 l2_factor: float,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(AddBiasLayer, self).__init__(trainable=trainable, name=name, **kwargs)
        self.global_average_score = global_average_score
        self.user_bias_score = None
        self.item_bias_score = None
        self.user_id_size = user_id_size
        self.item_id_size = item_id_size
        self.l2_factor = l2_factor

    def build(self, input_shape):
        self.user_bias_score = self.add_weight(
            name="user_bias_score",
            shape=(self.user_id_size,),
            dtype=self.dtype,
            regularizer=tf.keras.regularizers.l2(self.l2_factor))
        self.item_bias_score = self.add_weight(
            name="item_bias_score",
            shape=(self.item_id_size,),
            dtype=self.dtype,
            regularizer=tf.keras.regularizers.l2(self.l2_factor))

    def call(self, inputs, **kwargs):
        user_id, item_id = inputs
        user_bias = tf.nn.embedding_lookup(self.user_bias_score, user_id)
        item_bias = tf.nn.embedding_lookup(self.item_bias_score, item_id)
        return self.global_average_score + user_bias + item_bias


def main(_):
    model = svd_model(average_score=3.5, embedding_size=5, l2_factor=0.5)
    train_data = train_input_fn(batch_size=500)
    validate_data = test_input_fn(batch_size=1000)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.mean_squared_error,
                  metrics=tf.keras.metrics.RootMeanSquaredError())
    model.fit(x=train_data, validation_data=validate_data, epochs=1)
    tf.keras.utils.plot_model(model, to_file="./svd.png", rankdir="BT")


if __name__ == "__main__":
    app.run(main)

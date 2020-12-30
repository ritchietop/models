import tensorflow as tf
from absl import app
from data.movieLens import train_input_fn, test_input_fn


def svd_model(average_score, embedding_size, l2_factor, user_id_size=6040, item_id_size=3952):
    user_id = tf.keras.layers.Input(shape=(1,), name="user_id", dtype=tf.int64)
    movie_id = tf.keras.layers.Input(shape=(1,), name="movie_id", dtype=tf.int64)

    user_id_embedding_input = tf.keras.layers.Embedding(
        input_dim=user_id_size, output_dim=embedding_size,
        embeddings_regularizer=tf.keras.regularizers.l2(l2_factor))(user_id)
    user_bias_input = tf.keras.layers.Embedding(
        input_dim=user_id_size, output_dim=1,
        embeddings_regularizer=tf.keras.regularizers.l2(l2_factor))(user_id)
    movie_id_embedding_input = tf.keras.layers.Embedding(
        input_dim=item_id_size, output_dim=embedding_size,
        embeddings_regularizer=tf.keras.regularizers.l2(l2_factor))(movie_id)
    movie_bias_input = tf.keras.layers.Embedding(
        input_dim=item_id_size, output_dim=1, embeddings_regularizer=tf.keras.regularizers.l2(l2_factor))(movie_id)

    predict_no_bias = tf.keras.layers.Dot(axes=-1)(inputs=[user_id_embedding_input, movie_id_embedding_input])
    predict = tf.keras.layers.Add()(inputs=[user_bias_input, movie_bias_input, predict_no_bias])
    predict = tf.keras.layers.Lambda(function=lambda tensor: tf.nn.bias_add(tf.squeeze(tensor, axis=1),
                                                                            tf.constant(value=[average_score])),
                                     name="AddAverageScoreLayer")(predict)

    model = tf.keras.Model(inputs=[user_id, movie_id], outputs=[predict])
    return model


def main(_):
    model = svd_model(average_score=3.5, embedding_size=5, l2_factor=0.5)
    train_data = train_input_fn(batch_size=500)
    validate_data = test_input_fn(batch_size=1000)
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
                  loss=tf.keras.losses.mean_squared_error,
                  metrics=tf.keras.metrics.RootMeanSquaredError())
    model.fit(x=train_data, validation_data=validate_data, epochs=2)
    tf.keras.utils.plot_model(model, to_file="./svd.png", rankdir="BT")


if __name__ == "__main__":
    app.run(main)

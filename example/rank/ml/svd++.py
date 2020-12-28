import tensorflow as tf
from absl import app
import os


def svd_plus_plus_model(average_score, embedding_size, l2_factor, user_id_size, item_id_size):
    user_id_input = tf.keras.layers.Input(shape=(1,), name="user_id", dtype=tf.int32)
    item_id_input = tf.keras.layers.Input(shape=(1,), name="item_id", dtype=tf.int32)
    user_history_click_item_id_input = tf.keras.layers.Input(shape=(None,), name="user_history_click_item_id",
                                                             dtype=tf.int32, sparse=True)
    user_history_like_item_id_input = tf.keras.layers.Input(shape=(None,), name="user_history_like_item_id",
                                                            dtype=tf.int32, sparse=True)

    user_history_click_item_id_input_layer = tf.keras.layers.experimental.preprocessing.IntegerLookup(
        vocabulary=list(range(item_id_size)), mask_value=None)(user_history_click_item_id_input)
    user_history_like_item_id_input_layer = tf.keras.layers.experimental.preprocessing.IntegerLookup(
        vocabulary=list(range(item_id_size)), mask_value=None)(user_history_like_item_id_input)

    user_id_embedding_input_layer = tf.keras.layers.Embedding(
        input_dim=user_id_size, output_dim=embedding_size, embeddings_regularizer=tf.keras.regularizers.l2(l2_factor),
        name="UserIdEmbeddingLayer")(user_id_input)
    user_bias_input_layer = tf.keras.layers.Embedding(
        input_dim=user_id_size, output_dim=1, embeddings_regularizer=tf.keras.regularizers.l2(l2_factor),
        name="UserBiasLayer")(user_id_input)
    item_id_embedding_input_layer = tf.keras.layers.Embedding(
        input_dim=item_id_size, output_dim=embedding_size, embeddings_regularizer=tf.keras.regularizers.l2(l2_factor),
        name="ItemIdEmbeddingLayer")(item_id_input)
    item_bias_input_layer = tf.keras.layers.Embedding(
        input_dim=item_id_size, output_dim=1, embeddings_regularizer=tf.keras.regularizers.l2(l2_factor),
        name="ItemBiasLayer")(item_id_input)
    user_history_click_item_id_input_layer = tf.keras.layers.Embedding(
        input_dim=item_id_size, output_dim=embedding_size, embeddings_regularizer=tf.keras.regularizers.l2(l2_factor),
        name="UserHistoryClickItemIdLayer")(user_history_click_item_id_input_layer)
    user_history_like_item_id_input_layer = tf.keras.layers.Embedding(
        input_dim=item_id_size, output_dim=embedding_size, embeddings_regularizer=tf.keras.regularizers.l2(l2_factor),
        name="UserHistoryLikeItemLayer")(user_history_like_item_id_input_layer)

    user_embedding_input_layer = tf.keras.layers.Add()([user_id_embedding_input_layer,
                                                        user_history_click_item_id_input_layer,
                                                        user_history_like_item_id_input_layer])

    predict_no_bias = tf.keras.layers.Dot(axes=-1)([user_embedding_input_layer, item_id_embedding_input_layer])

    predict = tf.keras.layers.Add()([user_bias_input_layer, item_bias_input_layer, predict_no_bias])

    predict = tf.keras.layers.Lambda(function=lambda tensor: tf.nn.bias_add(tf.squeeze(tensor, axis=1),
                                                                            tf.constant(value=[average_score])),
                                     name="AddAverageScoreLayer")(predict)

    model = tf.keras.Model(inputs=[user_id_input, item_id_input], outputs=[predict])
    return model


def input_fn(path, batch_size, num_epochs=1, shuffle_buffer_size=None):
    return tf.data.experimental.make_csv_dataset(
        file_pattern=path,
        batch_size=batch_size,
        column_names=["user_id", "empty1", "item_id", "empty2", "rating", "empty3", "timestamp"],
        select_columns=["user_id", "item_id", "rating"],
        column_defaults=[0, 0, 0.0],
        label_name="rating",
        field_delim=":",
        use_quote_delim=True,
        na_value="null",
        header=False,
        num_epochs=num_epochs,
        shuffle=bool(shuffle_buffer_size),
        shuffle_buffer_size=shuffle_buffer_size,
        num_rows_for_inference=0,
        ignore_errors=False)


def main(_):
    model = svd_plus_plus_model(average_score=3.5, embedding_size=200, l2_factor=0.5, user_id_size=6040,
                                item_id_size=4000)
    data_path = os.path.abspath(__file__).replace("example/rank/ml/svd++.py", "data/movieLens/ml-1m/ratings.dat")
    train_data = input_fn(data_path, batch_size=500, shuffle_buffer_size=1000)
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                  loss=tf.keras.losses.mean_squared_error,
                  metrics=tf.keras.metrics.RootMeanSquaredError())
    model.fit(x=train_data, epochs=5)
    tf.keras.utils.plot_model(model, to_file="./svd++.png")


if __name__ == "__main__":
    app.run(main)

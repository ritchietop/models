import tensorflow as tf
from absl import app
from data.movieLens import train_input_fn, test_input_fn, user_occupation
from typing import List


def deep_fm_model(embedding_size: int, hidden_units: List[int], dropout: float, activation_fn=tf.keras.activations.relu):
    timestamp = tf.keras.layers.Input(shape=(1,), name="timestamp", dtype=tf.int64)
    gender = tf.keras.layers.Input(shape=(1,), name="gender", dtype=tf.string)
    age = tf.keras.layers.Input(shape=(1,), name="age", dtype=tf.int64)
    occupation = tf.keras.layers.Input(shape=(1,), name="occupation", dtype=tf.string)
    zip_code = tf.keras.layers.Input(shape=(1,), name="zip_code", dtype=tf.string)
    keywords = tf.keras.layers.Input(shape=(None,), name="keywords", dtype=tf.string, ragged=True)
    publish_year = tf.keras.layers.Input(shape=(1,), name="publishYear", dtype=tf.int64)
    categories = tf.keras.layers.Input(shape=(None,), name="categories", dtype=tf.string, ragged=True)

    # encode
    timestamp_hour_layer = tf.keras.layers.experimental.preprocessing.Discretization(bins=list(range(24)))(
        tf.keras.layers.Lambda(function=lambda tensor: tensor // 3600 % 24)(timestamp))
    timestamp_week_layer = tf.keras.layers.experimental.preprocessing.Discretization(bins=list(range(7)))(
        tf.keras.layers.Lambda(function=lambda tensor: tensor // 86400 % 7)(timestamp))
    gender_layer = tf.keras.layers.experimental.preprocessing.StringLookup(
        vocabulary=["F", "M"], mask_token=None, num_oov_indices=0)(gender)
    age_layer = tf.keras.layers.experimental.preprocessing.IntegerLookup(
        vocabulary=[1, 18, 25, 35, 45, 50, 56], mask_value=None, num_oov_indices=0)(age)
    occupation_layer = tf.keras.layers.experimental.preprocessing.StringLookup(
        vocabulary=list(user_occupation.values()), mask_token=None, num_oov_indices=0)(occupation)
    # 3439
    zip_code_layer = tf.keras.layers.experimental.preprocessing.Hashing(num_bins=10000)(zip_code)
    # 4862
    keywords_layer = tf.keras.layers.experimental.preprocessing.Hashing(num_bins=10000)(keywords)
    # 1919 ~ 2000
    publish_year_layer = tf.keras.layers.experimental.preprocessing.IntegerLookup(
        vocabulary=list(range(1919, 2001)), mask_value=None, num_oov_indices=0)(publish_year)
    categories_layer = tf.keras.layers.experimental.preprocessing.StringLookup(
        vocabulary=["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama",
                    "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War",
                    "Western"], mask_token=None, num_oov_indices=0)(categories)

    # lr
    timestamp_hour_lr_layer = tf.keras.layers.experimental.preprocessing.CategoryEncoding(
        max_tokens=24)(timestamp_hour_layer)
    timestamp_week_lr_layer = tf.keras.layers.experimental.preprocessing.CategoryEncoding(
        max_tokens=7)(timestamp_week_layer)
    gender_lr_layer = tf.keras.layers.experimental.preprocessing.CategoryEncoding(
        max_tokens=2, output_mode="binary")(gender_layer)
    age_lr_layer = tf.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens=7)(age_layer)
    occupation_lr_layer = tf.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens=21)(occupation_layer)
    # 3439
    zip_code_lr_layer = tf.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens=10000)(zip_code_layer)
    # 4862
    keywords_lr_layer = tf.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens=10000)(keywords_layer)
    # 1919 ~ 2000
    publish_year_lr_layer = tf.keras.layers.experimental.preprocessing.CategoryEncoding(
        max_tokens=82)(publish_year_layer)
    categories_lr_layer = tf.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens=18)(categories_layer)

    lr_input_layer = tf.keras.layers.Concatenate()(inputs=[
        timestamp_hour_lr_layer, timestamp_week_lr_layer, gender_lr_layer, age_lr_layer, occupation_lr_layer,
        zip_code_lr_layer, keywords_lr_layer, publish_year_lr_layer, categories_lr_layer
    ])
    lr_logits_layer = tf.keras.layers.Dense(units=1, name="LinearLayer")(lr_input_layer)

    # embedding_layers
    timestamp_hour_embed_layer = tf.keras.layers.Embedding(input_dim=24, output_dim=embedding_size)(timestamp_hour_layer)
    timestamp_week_embed_layer = tf.keras.layers.Embedding(input_dim=7, output_dim=embedding_size)(timestamp_week_layer)
    gender_embed_layer = tf.keras.layers.Embedding(input_dim=2, output_dim=embedding_size)(gender_layer)
    age_embed_layer = tf.keras.layers.Embedding(input_dim=7, output_dim=embedding_size)(age_layer)
    occupation_embed_layer = tf.keras.layers.Embedding(input_dim=21, output_dim=embedding_size)(occupation_layer)
    # 3439
    zip_code_embed_layer = tf.keras.layers.Embedding(input_dim=10000, output_dim=embedding_size)(zip_code_layer)
    # 4862
    keywords_embed_layer = tf.keras.layers.Embedding(input_dim=10000, output_dim=embedding_size)(keywords_layer)
    keywords_embed_layer = tf.keras.layers.Lambda(
        function=lambda tensor: tf.expand_dims(embedding_pooling(tensor), axis=1))(keywords_embed_layer)
    # 1919 ~ 2000
    publish_year_embed_layer = tf.keras.layers.Embedding(input_dim=82, output_dim=embedding_size)(publish_year_layer)
    categories_embed_layer = tf.keras.layers.Embedding(input_dim=18, output_dim=embedding_size)(categories_layer)
    categories_embed_layer = tf.keras.layers.Lambda(
        function=lambda tensor: tf.expand_dims(embedding_pooling(tensor), axis=1))(categories_embed_layer)

    all_embeddings_layer = tf.keras.layers.Concatenate(axis=1)(inputs=[
        timestamp_hour_embed_layer, timestamp_week_embed_layer,
        gender_embed_layer,
        age_embed_layer, occupation_embed_layer,
        zip_code_embed_layer, keywords_embed_layer, publish_year_embed_layer, categories_embed_layer
    ])

    fm_logits_layer = tf.keras.layers.Lambda(function=fm_cross, name="FmCrossLayer")(all_embeddings_layer)

    dnn_input_layer = tf.keras.layers.Flatten()(all_embeddings_layer)
    for hidden_unit in hidden_units:
        dnn_input_layer = tf.keras.layers.Dense(units=hidden_unit, activation=activation_fn)(dnn_input_layer)
        dnn_input_layer = tf.keras.layers.Dropout(rate=dropout)(dnn_input_layer)
    dnn_logits_layer = tf.keras.layers.Dense(units=1, activation=None)(dnn_input_layer)

    logits_layer = tf.keras.layers.Add()(inputs=[lr_logits_layer, fm_logits_layer, dnn_logits_layer])
    predict = tf.keras.layers.Lambda(function=lambda tensor: tf.nn.sigmoid(tensor), name="Sigmoid")(logits_layer)

    model = tf.keras.Model(inputs=[
        timestamp, gender, age, occupation, zip_code, keywords, publish_year, categories
    ], outputs=predict)

    return model


def fm_cross(embeddings):
    square_sum_tensor = tf.math.square(tf.math.reduce_sum(embeddings, axis=1))
    sum_square_tensor = tf.math.reduce_sum(tf.math.square(embeddings), axis=1)
    return 0.5 * tf.math.reduce_sum(square_sum_tensor - sum_square_tensor, axis=1, keepdims=True)


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


def main(_):
    model = deep_fm_model(embedding_size=10, hidden_units=[128, 128, 128], dropout=0.5)
    train_data = train_input_fn(batch_size=500)
    validate_data = test_input_fn(batch_size=1000)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.mean_squared_error,
                  metrics=tf.keras.metrics.RootMeanSquaredError())
    model.fit(x=train_data, validation_data=validate_data, epochs=1)
    tf.keras.utils.plot_model(model, to_file="./deep_fm.png", rankdir="BT")


if __name__ == "__main__":
    app.run(main)

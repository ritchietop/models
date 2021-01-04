import tensorflow as tf
from absl import app
from data.movieLens import train_input_fn, test_input_fn, user_occupation


def mlr_model(split_count: int):
    timestamp = tf.keras.layers.Input(shape=(1,), name="timestamp", dtype=tf.int64)
    gender = tf.keras.layers.Input(shape=(1,), name="gender", dtype=tf.string)
    age = tf.keras.layers.Input(shape=(1,), name="age", dtype=tf.int64)
    occupation = tf.keras.layers.Input(shape=(1,), name="occupation", dtype=tf.string)
    zip_code = tf.keras.layers.Input(shape=(1,), name="zip_code", dtype=tf.string)
    keywords = tf.keras.layers.Input(shape=(None,), name="keywords", dtype=tf.string, ragged=True)
    publish_year = tf.keras.layers.Input(shape=(1,), name="publishYear", dtype=tf.int64)
    categories = tf.keras.layers.Input(shape=(None,), name="categories", dtype=tf.string, ragged=True)

    timestamp_hour_layer = tf.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens=24)(
        tf.keras.layers.experimental.preprocessing.Discretization(bins=list(range(24)))(
            tf.keras.layers.Lambda(function=lambda tensor: tensor // 3600 % 24)(timestamp)))
    timestamp_week_layer = tf.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens=7)(
        tf.keras.layers.experimental.preprocessing.Discretization(bins=list(range(7)))(
            tf.keras.layers.Lambda(function=lambda tensor: tensor // 86400 % 7)(timestamp)))
    gender_layer = tf.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens=2, output_mode="binary")(
        tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=["F", "M"], mask_token=None)(gender))
    age_layer = tf.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens=7)(
        tf.keras.layers.experimental.preprocessing.IntegerLookup(
            vocabulary=[1, 18, 25, 35, 45, 50, 56], mask_value=None)(age))
    occupation_layer = tf.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens=21)(
        tf.keras.layers.experimental.preprocessing.StringLookup(
            vocabulary=list(user_occupation.values()), mask_token=None)(occupation))
    # 3439
    zip_code_layer = tf.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens=10000)(
        tf.keras.layers.experimental.preprocessing.Hashing(num_bins=10000)(zip_code))
    # 4862
    keywords_layer = tf.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens=10000)(
        tf.keras.layers.experimental.preprocessing.Hashing(num_bins=10000)(keywords))
    # 1919 ~ 2000
    publish_year_layer = tf.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens=82)(
        tf.keras.layers.experimental.preprocessing.IntegerLookup(
            vocabulary=list(range(1919, 2001)), mask_value=None)(publish_year))
    categories_layer = tf.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens=18)(
        tf.keras.layers.experimental.preprocessing.StringLookup(
            vocabulary=["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama",
                        "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War",
                        "Western"], mask_token=None)(categories))

    inputs = tf.keras.layers.Concatenate(axis=1)(inputs=[
        timestamp_hour_layer, timestamp_week_layer, gender_layer, age_layer, occupation_layer, zip_code_layer,
        keywords_layer, publish_year_layer, categories_layer
    ])

    # softmax
    softmax_layer = tf.keras.layers.Dense(units=split_count, activation=tf.keras.activations.softmax,
                                          name="SoftMax")(inputs)
    softmax_layer = tf.keras.layers.Lambda(
        function=lambda tensor: tensor / tf.math.reduce_sum(tensor, axis=1, keepdims=True))(softmax_layer)
    # sigmoid
    sigmoid_layer = tf.keras.layers.Dense(units=split_count, activation=tf.keras.activations.sigmoid,
                                          name="Sigmoid")(inputs)

    predict = tf.keras.layers.Multiply()(inputs=[softmax_layer, sigmoid_layer])
    predict = tf.keras.layers.Lambda(function=lambda tensor: tf.math.reduce_sum(tensor, axis=1), name="Sum")(predict)

    model = tf.keras.Model(inputs=[
        timestamp, gender, age, occupation, zip_code, keywords, publish_year, categories
    ], outputs=predict)

    return model


def main(_):
    model = mlr_model(split_count=5)
    train_data = train_input_fn(batch_size=500)
    validate_data = test_input_fn(batch_size=1000)
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
                  loss=tf.keras.losses.mean_squared_error,
                  metrics=tf.keras.metrics.RootMeanSquaredError())
    model.fit(x=train_data, validation_data=validate_data, epochs=1)
    tf.keras.utils.plot_model(model, to_file="./mlr.png", rankdir="BT")


if __name__ == "__main__":
    app.run(main)


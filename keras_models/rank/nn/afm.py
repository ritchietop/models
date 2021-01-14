import tensorflow as tf
from absl import app
from data.movieLens import train_input_fn, test_input_fn, user_occupation


def afm_model(embedding_size, l2_factor, hidden_unit, dropout):
    timestamp = tf.keras.layers.Input(shape=(1,), name="timestamp", dtype=tf.int64)
    gender = tf.keras.layers.Input(shape=(1,), name="gender", dtype=tf.string)
    age = tf.keras.layers.Input(shape=(1,), name="age", dtype=tf.int64)
    occupation = tf.keras.layers.Input(shape=(1,), name="occupation", dtype=tf.string)
    zip_code = tf.keras.layers.Input(shape=(1,), name="zip_code", dtype=tf.string)
    keywords = tf.keras.layers.Input(shape=(None,), name="keywords", dtype=tf.string, ragged=True)
    publish_year = tf.keras.layers.Input(shape=(1,), name="publishYear", dtype=tf.int64)
    categories = tf.keras.layers.Input(shape=(None,), name="categories", dtype=tf.string, ragged=True)

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
    lr_input_layer = tf.keras.layers.Dense(units=1, name="LinearLayer", use_bias=True,
                                           kernel_regularizer=tf.keras.regularizers.l2(l2_factor))(lr_input_layer)

    # fm
    embedding_pooling_layer = tf.keras.layers.Lambda(lambda tensor: embedding_pooling(tensor), name="EmbeddingPooling")
    timestamp_hour_fm_layer = tf.keras.layers.Embedding(input_dim=24, output_dim=embedding_size)(timestamp_hour_layer)
    timestamp_hour_fm_layer = embedding_pooling_layer(timestamp_hour_fm_layer)
    timestamp_week_fm_layer = tf.keras.layers.Embedding(input_dim=7, output_dim=embedding_size)(timestamp_week_layer)
    timestamp_week_fm_layer = embedding_pooling_layer(timestamp_week_fm_layer)
    gender_fm_layer = tf.keras.layers.Embedding(input_dim=2, output_dim=embedding_size)(gender_layer)
    gender_fm_layer = embedding_pooling_layer(gender_fm_layer)
    age_fm_layer = tf.keras.layers.Embedding(input_dim=7, output_dim=embedding_size)(age_layer)
    age_fm_layer = embedding_pooling_layer(age_fm_layer)
    occupation_fm_layer = tf.keras.layers.Embedding(input_dim=21, output_dim=embedding_size)(occupation_layer)
    occupation_fm_layer = embedding_pooling_layer(occupation_fm_layer)
    # 3439
    zip_code_fm_layer = tf.keras.layers.Embedding(input_dim=10000, output_dim=embedding_size)(zip_code_layer)
    zip_code_fm_layer = embedding_pooling_layer(zip_code_fm_layer)
    # 4862
    keywords_fm_layer = tf.keras.layers.Embedding(input_dim=10000, output_dim=embedding_size)(keywords_layer)
    keywords_fm_layer = embedding_pooling_layer(keywords_fm_layer)
    # 1919 ~ 2000
    publish_year_fm_layer = tf.keras.layers.Embedding(input_dim=82, output_dim=embedding_size)(publish_year_layer)
    publish_year_fm_layer = embedding_pooling_layer(publish_year_fm_layer)
    categories_fm_layer = tf.keras.layers.Embedding(input_dim=18, output_dim=embedding_size)(categories_layer)
    categories_fm_layer = embedding_pooling_layer(categories_fm_layer)

    # fm pair-wise interaction layer
    pair_wise_interaction_layer = tf.keras.layers.Lambda(
        function=pair_wise_interaction_layer_fn, name="PairWiseInteractionLayer")(
        inputs=[timestamp_hour_fm_layer, timestamp_week_fm_layer, gender_fm_layer, age_fm_layer, occupation_fm_layer,
                zip_code_fm_layer, keywords_fm_layer, publish_year_fm_layer, categories_fm_layer])
    # dropout layer
    pair_wise_interaction_layer = tf.keras.layers.Dropout(rate=dropout)(pair_wise_interaction_layer)
    # attention layer
    attention_fm_layer = AttentionLayer(hidden_unit=hidden_unit, l2_factor=l2_factor)(pair_wise_interaction_layer)

    logits = tf.keras.layers.Add(name="LogitsLayer")(inputs=[lr_input_layer, attention_fm_layer])
    predict = tf.keras.layers.Lambda(function=lambda tensor: tf.nn.sigmoid(tensor), name="SigmoidLayer")(logits)

    model = tf.keras.models.Model(inputs=[
        timestamp, gender, age, occupation, zip_code, keywords, publish_year, categories
    ], outputs=[predict])

    return model


# tensor: [batch_size, record_count, embedding_size]
def embedding_pooling(tensor, combiner: str = "sqrtn"):
    tensor_sum = tf.math.reduce_sum(tensor, axis=1)  # batch_size * embedding_size
    if combiner == "sum":
        return tensor_sum
    if isinstance(tensor, tf.RaggedTensor):
        row_lengths = tf.expand_dims(tensor.row_lengths(axis=1), axis=1)  # batch_size * 1
        row_lengths = tf.math.maximum(tf.ones_like(row_lengths), row_lengths)
    elif isinstance(tensor, tf.Tensor):
        row_lengths = tensor.shape[1]
    else:
        raise ValueError("Only support Tensor or RaggedTensor.")
    row_lengths = tf.cast(row_lengths, dtype=tf.float32)
    if combiner == "mean":
        return tensor_sum / row_lengths
    if combiner == "sqrtn":
        return tensor_sum / tf.math.sqrt(row_lengths)


# output: [batch_size, m * (m - 1) / 2, embedding_size]
def pair_wise_interaction_layer_fn(tensors):
    interaction_tensors = []
    for i in range(len(tensors)):
        for j in range(i):
            interaction_tensors.append(tf.math.multiply(tensors[j], tensors[i]))
    return tf.stack(interaction_tensors, axis=1)


class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_unit, l2_factor, trainable=True, name=None, **kwargs):
        super(AttentionLayer, self).__init__(trainable=trainable, name=name, **kwargs)
        self.hidden_unit = hidden_unit
        self.dense_layer = tf.keras.layers.Dense(units=hidden_unit, activation=tf.keras.activations.relu,
                                                 kernel_regularizer=tf.keras.regularizers.l2(l2_factor))
        self.h = None
        self.p = None

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        self.h = self.add_weight(name="attention_h", shape=(self.hidden_unit, 1), dtype=tf.float32)
        self.p = self.add_weight(name="attention_p", shape=(input_shape[-1], 1), dtype=tf.float32)

    def call(self, inputs, **kwargs):
        interaction_tensor = inputs  # batch * [m * (m - 1) / 2] * embedding
        inputs = self.dense_layer(inputs)  # batch * [m * (m - 1) / 2] * hidden_unit
        inputs = tf.matmul(inputs, self.h)  # batch * [m * (m - 1) / 2] * 1
        inputs = tf.math.exp(inputs)  # batch * [m * (m - 1) / 2] * 1
        normalize_weights = inputs / tf.math.reduce_sum(inputs, axis=1, keepdims=True)  # batch * [m * (m - 1) / 2] * 1
        weighted_interaction_tensor = interaction_tensor * normalize_weights  # batch * [m * (m - 1) / 2] * embedding
        weighted_interaction_tensor = tf.math.reduce_sum(weighted_interaction_tensor, axis=1)  # batch * embedding
        return tf.matmul(weighted_interaction_tensor, self.p)


def main(_):
    model = afm_model(embedding_size=10, l2_factor=0.5, hidden_unit=10, dropout=0.5)
    train_data = train_input_fn(batch_size=500)
    validate_data = test_input_fn(batch_size=1000)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.mean_squared_error,
                  metrics=tf.keras.metrics.RootMeanSquaredError())
    model.fit(x=train_data, validation_data=validate_data, epochs=1)
    tf.keras.utils.plot_model(model, to_file="./afm.png", rankdir="BT")


if __name__ == "__main__":
    app.run(main)

import tensorflow as tf
from absl import app
from data.movieLens_graph import input_fn


def item2vec_model(embedding_size, num_sampled, item_id_size=3953):
    target_movie_id = tf.keras.layers.Input(shape=(1,), name="target_movie_id", dtype=tf.int64)
    movie_id = tf.keras.layers.Input(shape=(1,), name="movie_id", dtype=tf.int64)

    item2vec_layer = Item2VecLayer(embedding_size, item_id_size, num_sampled)(inputs=[movie_id, target_movie_id])

    model = tf.keras.Model(inputs=[target_movie_id, movie_id], outputs=item2vec_layer)

    return model


class Item2VecLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_size, item_id_size, num_sampled, num_true=1, trainable=True, name=None, **kwargs):
        super(Item2VecLayer, self).__init__(trainable=trainable, name=name, **kwargs)
        self.embedding_size = embedding_size
        self.item_id_size = item_id_size
        self.num_sampled = num_sampled
        self.num_true = num_true
        self.front_item_embeddings = None
        self.after_item_embeddings = None
        self.after_item_bias = None

    def build(self, input_shape):
        self.front_item_embeddings = self.add_weight(
            name="front_item_embedding",
            shape=(self.item_id_size, self.embedding_size),
            dtype=self.dtype,
            trainable=True)
        self.after_item_embeddings = self.add_weight(
            name="after_item_embedding",
            shape=(self.item_id_size, self.embedding_size),
            dtype=self.dtype,
            trainable=True)
        self.after_item_bias = self.add_weight(
            name="after_item_bias",
            shape=(self.item_id_size,),
            dtype=self.dtype,
            trainable=True)

    def call(self, inputs, training=None):
        movie_id, target_movie_id = inputs
        movie_id_tensor = tf.squeeze(tf.nn.embedding_lookup(self.front_item_embeddings, movie_id), axis=1)
        if training:
            loss = tf.nn.sampled_softmax_loss(
                weights=self.after_item_embeddings,
                biases=self.after_item_bias,
                labels=target_movie_id,
                inputs=movie_id_tensor,
                num_sampled=self.num_sampled,
                num_classes=self.item_id_size,
                num_true=self.num_true)
        else:
            logits = tf.matmul(movie_id_tensor, self.after_item_embeddings, transpose_b=True)
            logits = tf.nn.bias_add(logits, self.after_item_bias)
            labels = tf.one_hot(indices=target_movie_id, depth=self.item_id_size)
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        return loss


def main(_):
    model = item2vec_model(embedding_size=128, num_sampled=10)
    train_data = input_fn(batch_size=500, shuffle_buffer_size=1000)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.mean_squared_error,
                  metrics=tf.keras.metrics.RootMeanSquaredError())
    model.fit(x=train_data, epochs=1)
    tf.keras.utils.plot_model(model, to_file="./item2vec.png", rankdir="BT")


if __name__ == "__main__":
    app.run(main)

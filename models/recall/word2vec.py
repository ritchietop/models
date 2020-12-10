import tensorflow as tf


class Word2VecModel(tf.keras.Model):
    def __init__(self, word_embedding_size: int, word_hash_size: int, num_sampled: int, num_true: int,
                 name=None, **kwargs):
        super(Word2VecModel, self).__init__(name=name, **kwargs)
        self.word_embedding_size = word_embedding_size
        self.word_hash_size = word_hash_size
        self.num_sampled = num_sampled
        self.num_true = num_true
        self.input_layer_embeddings = None
        self.output_layer_embeddings = None
        self.output_layer_bias = None

    def build(self, input_shape):
        self.input_layer_embeddings = self.add_weight(
            name="input_layer_embeddings",
            shape=(self.word_hash_size, self.word_embedding_size),
            dtype=self.dtype,
            trainable=True)
        self.output_layer_embeddings = self.add_weight(
            name="output_layer_embeddings",
            shape=(self.word_hash_size, self.word_embedding_size),
            dtype=self.dtype,
            trainable=True)
        self.output_layer_bias = self.add_weight(
            name="output_layer_bias",
            shape=(self.word_hash_size,),
            dtype=self.dtype,
            trainable=True)

    def get_config(self):
        config = {
            "word_embedding_size": self.word_embedding_size,
            "word_hash_size": self.word_hash_size,
            "num_sampled": self.num_sampled,
            "num_true": self.num_true
        }
        base_config = super(Word2VecModel, self).get_config()
        return {**base_config, **config}

    def get_word_one_hot_encode(self, word):
        return tf.strings.to_hash_bucket_fast(word, self.word_hash_size)

    def call(self, inputs, training=None, mask=None):
        input_word, output_word = inputs
        input_word_id_tensor = self.get_word_one_hot_encode(input_word)
        input_word_embedding = tf.nn.embedding_lookup(params=self.input_layer_embeddings, ids=input_word_id_tensor)
        input_word_embedding = tf.math.reduce_sum(input_word_embedding, axis=1)
        output_word_id_tensor = self.get_word_one_hot_encode(output_word)
        if training:
            loss = tf.nn.sampled_softmax_loss(
                weights=self.output_layer_embeddings,
                biases=self.output_layer_bias,
                labels=output_word_id_tensor,
                inputs=input_word_embedding,
                num_classes=self.word_hash_size,
                num_true=1)
        else:
            logits = tf.matmul(input_word_embedding, self.output_layer_embeddings, transpose_b=True)
            logits = tf.nn.bias_add(logits, self.output_layer_bias)
            labels = tf.one_hot(output_word_id_tensor, depth=self.word_hash_size)
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        return loss

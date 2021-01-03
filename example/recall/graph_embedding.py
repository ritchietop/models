import tensorflow as tf
from tensorflow.python import context
from tensorflow.python.distribute.sharded_variable import ShardedVariable
from tensorflow.python.data.experimental.ops.readers import make_tf_record_dataset
from absl import app, flags
#from models.utils import distribution_utils

'''
tensorflow==2.3.1
'''


class SparseEmbeddingLayer(tf.keras.layers.Embedding):
    def __init__(self, combiner="sum", **kwargs):
        super(SparseEmbeddingLayer, self).__init__(**kwargs)
        self.combiner = combiner

    def call(self, inputs):
        if isinstance(inputs, list):
            id_inputs = inputs[0]
            weight_inputs = inputs[1] if len(inputs) == 2 else None
        else:
            id_inputs = inputs
            weight_inputs = None
        dtype = tf.keras.backend.dtype(id_inputs)
        if dtype != 'int32' and dtype != 'int64':
            id_inputs = tf.cast(id_inputs, 'int32')
        if isinstance(self.embeddings, ShardedVariable):
            embedding_weights = self.embeddings.variables
        else:
            embedding_weights = self.embeddings
        out = tf.nn.safe_embedding_lookup_sparse(embedding_weights=embedding_weights, sparse_ids=id_inputs,
                                                 sparse_weights=weight_inputs, combiner=self.combiner)
        return out


#@tf.function
def weighted_item_embedding_func(inputs):
    item_attention_tensor, item_embedding_stack_tensor = inputs
    item_attention_tensor = tf.math.exp(tf.squeeze(item_attention_tensor, axis=1))
    item_attention_tensor_std = item_attention_tensor / tf.reduce_sum(item_attention_tensor, axis=1, keepdims=True)
    item_weighted_embedding_tensor = tf.matmul(item_embedding_stack_tensor,
                                               tf.expand_dims(item_attention_tensor_std, axis=2))
    item_weighted_embedding_tensor = tf.squeeze(item_weighted_embedding_tensor, axis=2)
    return item_weighted_embedding_tensor


class SampledSoftMaxLossLayer(tf.keras.layers.Layer):
    def __init__(self,
                 item_id_hash_size: int,
                 embedding_size: int,
                 embeddings_initializer,
                 embeddings_regularizer,
                 num_sampled: int,
                 num_true: int,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(SampledSoftMaxLossLayer, self).__init__(trainable=trainable, name=name, **kwargs)
        self.item_id_hash_size = item_id_hash_size
        self.embedding_size = embedding_size
        self.embeddings_initializer = embeddings_initializer
        self.embeddings_regularizer = embeddings_regularizer
        self.num_sampled = num_sampled
        self.num_true = num_true
        self.embedding_weights = None
        self.biases = None

    def get_config(self):
        config = {
            "item_id_hash_size": self.item_id_hash_size,
            "embedding_size": self.embedding_size,
            "embedding_initializer": self.embeddings_initializer,
            "embedding_regularizer": self.embeddings_regularizer,
            "num_sampled": self.num_sampled,
            "num_true": self.num_true
        }
        base_config = super(SampledSoftMaxLossLayer, self).get_config()
        return {**base_config, **config}

    def build(self, input_shape):
        self.embedding_weights = self.add_weight(
            name="embedding_weights",
            shape=(self.item_id_hash_size, self.embedding_size),
            dtype=tf.float32,
            initializer=self.embeddings_initializer,
            regularizer=self.embeddings_regularizer,
            trainable=self.trainable)
        self.biases = self.add_weight(
            name="biases",
            shape=(self.item_id_hash_size,),
            initializer=tf.keras.initializers.zeros())

    def call(self, inputs, **kwargs):
        target_id_tensor, item_embeddings = inputs
        sampled_loss = tf.nn.sampled_softmax_loss(
            weights=self.embedding_weights,
            biases=self.biases,
            labels=target_id_tensor,
            inputs=item_embeddings,
            num_sampled=self.num_sampled,
            num_classes=self.item_id_hash_size,
            num_true=self.num_true,
            sampled_values=tf.random.uniform_candidate_sampler(
                true_classes=target_id_tensor,
                num_true=self.num_true,
                num_sampled=self.num_sampled,
                unique=True,
                range_max=self.item_id_hash_size))
        return sampled_loss


#@tf.function
def parse_record(tensor):
    feature_schema = {
        "item_id": tf.io.FixedLenFeature(shape=(1,), dtype=tf.string, default_value=""),
        "item_category": tf.io.VarLenFeature(dtype=tf.string),
        "item_broken": tf.io.FixedLenFeature(shape=(1,), dtype=tf.string, default_value=""),
        "item_copyright": tf.io.FixedLenFeature(shape=(1,), dtype=tf.string, default_value=""),
        "item_channel": tf.io.FixedLenFeature(shape=(1,), dtype=tf.string, default_value=""),
        "item_finished": tf.io.FixedLenFeature(shape=(1,), dtype=tf.string, default_value=""),
        "item_word_count": tf.io.FixedLenFeature(shape=(1,), dtype=tf.int64, default_value=0),
        "item_chapter_count": tf.io.FixedLenFeature(shape=(1,), dtype=tf.int64, default_value=0),
        "item_tags": tf.io.VarLenFeature(dtype=tf.string),
        "target_id": tf.io.FixedLenFeature(shape=(1,), dtype=tf.string, default_value="")
    }
    return tf.io.parse_example(tensor, feature_schema)


def graph_embedding_model(embedding_size, item_id_hash_size, item_category_file, item_broken_list,
                          item_copyright_file, item_channel_list, item_finished_list, item_word_count_boundary,
                          item_chapter_count_boundary, item_tags_file, num_sampled, embeddings_initializer,
                          embeddings_regularizer):
    example_input = tf.keras.layers.Input(shape=(), name="input_example_tensor", dtype=tf.string)
    example_tensors = tf.keras.layers.Lambda(parse_record, name="parse_layer")(example_input)

    item_id_input = example_tensors["item_id"]
    target_id_input = example_tensors["target_id"]
    item_category_input = example_tensors["item_category"]
    item_broken_input = example_tensors["item_broken"]
    item_copyright_input = example_tensors["item_copyright"]
    item_channel_input = example_tensors["item_channel"]
    item_finished_input = example_tensors["item_finished"]
    item_word_count_input = example_tensors["item_word_count"]
    item_chapter_count_input = example_tensors["item_chapter_count"]
    item_tags_input = example_tensors["item_tags"]

    # item_id_input = tf.keras.layers.Input(shape=(1,), name="item_id", dtype=tf.string, tensor=item_id_input)
    # target_id_input = tf.keras.layers.Input(shape=(1,), name="target_id", dtype=tf.string, tensor=target_id_input)
    # item_category_input = tf.keras.layers.Input(shape=(None,), name="item_category", dtype=tf.string, sparse=True, tensor=item_category_input)
    # item_broken_input = tf.keras.layers.Input(shape=(1,), name="item_broken", dtype=tf.string, tensor=item_broken_input)
    # item_copyright_input = tf.keras.layers.Input(shape=(1,), name="item_copyright", dtype=tf.string, tensor=item_copyright_input)
    # item_channel_input = tf.keras.layers.Input(shape=(1,), name="item_channel", dtype=tf.string, tensor=item_channel_input)
    # item_finished_input = tf.keras.layers.Input(shape=(1,), name="item_finished", dtype=tf.string, tensor=item_finished_input)
    # item_word_count_input = tf.keras.layers.Input(shape=(1,), name="item_word_count", dtype=tf.int64, tensor=item_word_count_input)
    # item_chapter_count_input = tf.keras.layers.Input(shape=(1,), name="item_chapter_count", dtype=tf.int64, tensor=item_chapter_count_input)
    # item_tags_input = tf.keras.layers.Input(shape=(None,), name="item_tags", dtype=tf.string, sparse=True, tensor=item_tags_input)

    item_id_preprocess_layer = tf.keras.layers.experimental.preprocessing.Hashing(num_bins=item_id_hash_size,
                                                                                  name="item_id_hash_layer")
    item_id_tensor = item_id_preprocess_layer(item_id_input)
    target_id_tensor = item_id_preprocess_layer(target_id_input)
    item_category_layer = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=item_category_file)
    item_category_tensor = item_category_layer(item_category_input)
    item_broken_layer = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=item_broken_list)
    item_broken_tensor = item_broken_layer(item_broken_input)
    item_copyright_layer = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=item_copyright_file)
    item_copyright_tensor = item_copyright_layer(item_copyright_input)
    item_channel_layer = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=item_channel_list)
    item_channel_tensor = item_channel_layer(item_channel_input)
    item_finished_layer = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=item_finished_list)
    item_finished_tensor = item_finished_layer(item_finished_input)
    item_word_count_layer = tf.keras.layers.experimental.preprocessing.Discretization(bins=item_word_count_boundary)
    item_word_count_tensor = item_word_count_layer(item_word_count_input)
    item_chapter_count_layer = tf.keras.layers.experimental.preprocessing.Discretization(
        bins=item_chapter_count_boundary)
    item_chapter_count_tensor = item_chapter_count_layer(item_chapter_count_input)
    item_tags_layer = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=item_tags_file)
    item_tags_tensor = item_tags_layer(item_tags_input)

    tensor_squeeze_layer = tf.keras.layers.Lambda(function=lambda tensor: tf.squeeze(tensor, axis=1),
                                                  name="squeeze_layer")

    item_id_embedding_tensor = tensor_squeeze_layer(tf.keras.layers.Embedding(
        input_dim=item_id_hash_size, output_dim=embedding_size,
        embeddings_initializer=embeddings_initializer,
        embeddings_regularizer=embeddings_regularizer)(item_id_tensor))
    item_category_embedding_tensor = SparseEmbeddingLayer(
        input_dim=item_category_layer.vocab_size(), output_dim=embedding_size, combiner="sqrtn",
        embeddings_initializer=embeddings_initializer,
        embeddings_regularizer=embeddings_regularizer)(item_category_tensor)
    item_broken_embedding_tensor = tensor_squeeze_layer(tf.keras.layers.Embedding(
        input_dim=item_broken_layer.vocab_size(), output_dim=embedding_size,
        embeddings_initializer=embeddings_initializer,
        embeddings_regularizer=embeddings_regularizer)(item_broken_tensor))
    item_copyright_embedding_tensor = tensor_squeeze_layer(tf.keras.layers.Embedding(
        input_dim=item_copyright_layer.vocab_size(), output_dim=embedding_size,
        embeddings_initializer=embeddings_initializer,
        embeddings_regularizer=embeddings_regularizer)(item_copyright_tensor))
    item_channel_embedding_tensor = tensor_squeeze_layer(tf.keras.layers.Embedding(
        input_dim=item_channel_layer.vocab_size(), output_dim=embedding_size,
        embeddings_initializer=embeddings_initializer,
        embeddings_regularizer=embeddings_regularizer)(item_channel_tensor))
    item_finished_embedding_tensor = tensor_squeeze_layer(tf.keras.layers.Embedding(
        input_dim=item_finished_layer.vocab_size(), output_dim=embedding_size,
        embeddings_initializer=embeddings_initializer,
        embeddings_regularizer=embeddings_regularizer)(item_finished_tensor))
    item_word_count_embedding_tensor = tensor_squeeze_layer(tf.keras.layers.Embedding(
        input_dim=len(item_word_count_boundary) + 1, output_dim=embedding_size,
        embeddings_initializer=embeddings_initializer,
        embeddings_regularizer=embeddings_regularizer)(item_word_count_tensor))
    item_chapter_count_embedding_tensor = tensor_squeeze_layer(tf.keras.layers.Embedding(
        input_dim=len(item_chapter_count_boundary) + 1, output_dim=embedding_size,
        embeddings_initializer=embeddings_initializer,
        embeddings_regularizer=embeddings_regularizer)(item_chapter_count_tensor))
    item_tags_embedding_tensor = SparseEmbeddingLayer(
        input_dim=item_tags_layer.vocab_size(), output_dim=embedding_size, combiner="sqrtn",
        embeddings_initializer=embeddings_initializer,
        embeddings_regularizer=embeddings_regularizer)(item_tags_tensor)

    item_feature_tensors = [item_id_embedding_tensor, item_category_embedding_tensor,
                            item_broken_embedding_tensor, item_copyright_embedding_tensor,
                            item_channel_embedding_tensor, item_finished_embedding_tensor,
                            item_word_count_embedding_tensor, item_chapter_count_embedding_tensor,
                            item_tags_embedding_tensor]
    item_embedding_stack_tensor = tf.keras.layers.Lambda(
        function=lambda inputs: tf.stack(inputs, axis=2),
        name="ItemEmbeddingStackLayer")(inputs=item_feature_tensors)

    item_weights_tensor = tf.keras.layers.Embedding(input_dim=item_id_hash_size, output_dim=len(item_feature_tensors),
                                                    embeddings_initializer=embeddings_initializer,
                                                    embeddings_regularizer=embeddings_regularizer,
                                                    name="ItemEmbeddingWeightsLayer")(item_id_tensor)

    item_weighted_embedding_tensor = tf.keras.layers.Lambda(
        function=weighted_item_embedding_func,
        name="ItemEmbeddingAttentionLayer")(inputs=[item_weights_tensor, item_embedding_stack_tensor])

    sampled_loss = SampledSoftMaxLossLayer(item_id_hash_size=item_id_hash_size,
                                           embedding_size=embedding_size,
                                           embeddings_initializer=embeddings_initializer,
                                           embeddings_regularizer=embeddings_regularizer,
                                           num_sampled=num_sampled,
                                           num_true=1)(
        inputs=[target_id_tensor, item_weighted_embedding_tensor])

    return tf.keras.Model(inputs=example_input, outputs=sampled_loss)


#@tf.function
def loss(y_true, y_pred):
    return tf.math.reduce_mean(y_pred)


flags.DEFINE_string("data_path", "/Users/ritchie/Projects/GitHub/models/examples/data/part-r-00002", "")
flags.DEFINE_integer("batch_size", 3, "")
flags.DEFINE_integer("shuffle_buffer_size", 3000, "")
flags.DEFINE_integer("embedding_size", 128, "")
flags.DEFINE_integer("item_id_hash_size", 100000, "")
flags.DEFINE_string("item_category_file", "/Users/ritchie/Projects/GitHub/models/models/copyrights", "")
flags.DEFINE_string("item_copyright_file", "/Users/ritchie/Projects/GitHub/models/models/item_category_list", "")
flags.DEFINE_string("item_tags_file", "/Users/ritchie/Projects/GitHub/models/models/item_tag_list", "")
flags.DEFINE_integer("sampled_num", 10, "")
flags.DEFINE_float("l2_factor", 1.0, "")
flags.DEFINE_integer("num_epochs", 2, "")
flags.DEFINE_float("learning_rate", 0.01, "")
flags.DEFINE_integer("lr_decay_steps", 10, "")
flags.DEFINE_float("lr_decay_rate", 0.95, "")
flags.DEFINE_string("model_checkpoint_path", "/Users/ritchie/Projects/GitHub/models/examples/data", "")
flags.DEFINE_string("distribution_strategy", "one_device", "")
flags.DEFINE_string("all_reduce_alg", None, "")

FLAGS = flags.FLAGS


def main(_):
    tf.config.optimizer.set_jit(True)
    tf.config.set_soft_device_placement(enabled=True)
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    try:
        for physical_device in physical_devices:
            tf.config.experimental.set_memory_growth(physical_device, True)
            print(str(physical_device) + " set memory growth success.")
    except _:
        pass

    data = make_tf_record_dataset(
        file_pattern=FLAGS.data_path,
        batch_size=FLAGS.batch_size,
        num_epochs=1,
        shuffle=bool(FLAGS.shuffle_buffer_size),
        shuffle_buffer_size=FLAGS.shuffle_buffer_size,
        drop_final_batch=False).take(10)
    data = data.map(lambda tensor: ({"input_example": tensor},
                                    tf.ones(shape=(FLAGS.batch_size, 1), dtype=tf.float32)))

    # strategy = distribution_utils.get_distribution_strategy(FLAGS.distribution_strategy,
    #                                                         num_gpus=context.num_gpus(),
    #                                                         all_reduce_alg=FLAGS.all_reduce_alg)

    #with strategy.scope():
    model = graph_embedding_model(
        embedding_size=FLAGS.embedding_size,
        item_id_hash_size=FLAGS.item_id_hash_size,
        item_category_file=FLAGS.item_category_file,
        item_broken_list=["True", "False"],
        item_copyright_file=FLAGS.item_copyright_file,
        item_channel_list=["1", "2", "1,2"],
        item_finished_list=["True", "False"],
        item_word_count_boundary=[1, 1_0000, 10_0000, 50_0000, 100_0000, 300_0000, 500_0000, 700_0000, 1000_0000],
        item_chapter_count_boundary=[1, 50, 100, 150, 200, 250, 300, 350, 450, 550, 750, 2000],
        item_tags_file=FLAGS.item_tags_file,
        num_sampled=FLAGS.sampled_num,
        embeddings_initializer=None,
        embeddings_regularizer=tf.keras.regularizers.l2(FLAGS.l2_factor))
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=FLAGS.learning_rate,
                                                                   decay_steps=FLAGS.lr_decay_steps,
                                                                   decay_rate=FLAGS.lr_decay_rate)
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, run_eagerly=False)

    model.fit(x=data, epochs=FLAGS.num_epochs,)
              # callbacks=[tf.keras.callbacks.TensorBoard(log_dir=FLAGS.model_checkpoint_path + "/tensorboard"),
              #            tf.keras.callbacks.ModelCheckpoint(
              #                filepath=FLAGS.model_checkpoint_path + "/model-{epoch:02d}/",
              #                monitor="loss", save_freq="epoch",
              #                mode="min", save_weights_only=False, save_best_only=False)])
    tf.saved_model.save(model, export_dir=FLAGS.model_checkpoint_path + "/model")

if __name__ == "__main__":
    model = tf.saved_model.load("/data/model/")
    print(model)
    #app.run(main)

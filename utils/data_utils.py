import tensorflow as tf
from tensorflow.python.data.ops.readers import ParallelInterleaveDataset


def make_batched_features_dataset(file_pattern,
                                  context_schema,
                                  sequence_schema,
                                  batch_size,
                                  num_epochs,
                                  shuffle_buffer_size,
                                  label_key,
                                  reader_num_threads=1,
                                  parser_num_threads=2,
                                  sloppy_ordering=False):
    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=bool(shuffle_buffer_size))
    if reader_num_threads == tf.data.AUTOTUNE:
        dataset = dataset.interleave(lambda filename: tf.data.TFRecordDataset(filename),
                                     num_parallel_calls=reader_num_threads)
    else:
        def apply_fn(dataset):
            return ParallelInterleaveDataset(dataset, lambda filename: tf.data.TFRecordDataset(filename),
                                             cycle_length=reader_num_threads, block_length=1, sloppy=sloppy_ordering,
                                             buffer_output_elements=None, prefetch_input_elements=None)

        dataset = dataset.apply(apply_fn)
    if shuffle_buffer_size is not None:
        dataset = dataset.shuffle(shuffle_buffer_size)
    if num_epochs != 1:
        dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size, drop_remainder=bool(shuffle_buffer_size) or num_epochs is None)

    def parse_func(serialized):
        context_features, sequence_features, features_length = tf.io.parse_sequence_example(
            serialized, context_features=context_schema, sequence_features=sequence_schema)
        return {**context_features, **sequence_features}

    dataset = dataset.map(parse_func, num_parallel_calls=parser_num_threads)
    if label_key is not None:
        if label_key not in context_schema:
            raise ValueError("The 'label_key' provided (%r) must be one of the 'features' keys." % label_key)
        dataset = dataset.map(lambda x: (x, x.pop(label_key)))
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

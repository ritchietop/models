import tensorflow as tf
import os
import random
import numpy as np
from collections import defaultdict
from data.movieLens import load_rating_data, load_movie_data


def alias_sample(neighbors):
    n = len(neighbors)
    U = np.array([prob for _, prob in neighbors]) * n
    K = [i for i in range(n)]
    over_full, under_full = [], []
    for i, U_i in enumerate(U):
        if U_i > 1:
            over_full.append(i)
        elif U_i < 1:
            under_full.append(i)
    while len(over_full) and len(under_full):
        i, j = over_full.pop(), under_full.pop()
        K[j] = i
        U[i] = U[i] - (1 - U[j])
        if U[i] > 1:
            over_full.append(i)
        elif U[i] < 1:
            under_full.append(i)
    id = np.random.randint(n)
    if U[id] > np.random.uniform(0, 1):
        return neighbors[id][0]
    else:
        return neighbors[K[id]][0]


def load_pair_movies_graph(rating_max_gap_second, is_direct: bool = False):
    pair_movies = defaultdict(lambda: defaultdict(int))
    rating_data = load_rating_data()
    for ratings in rating_data.values():
        sorted_ratings = sorted(list(ratings.items()), key=lambda item: item[1]["timestamp"])
        for i in range(len(sorted_ratings)):
            if i == 0:
                continue
            front_movie_id, front_movie_timestamp = sorted_ratings[i-1][0], sorted_ratings[i-1][1]["timestamp"]
            after_movie_id, after_movie_timestamp = sorted_ratings[i][0], sorted_ratings[i][1]["timestamp"]
            if after_movie_timestamp - front_movie_timestamp <= rating_max_gap_second:
                pair_movies[front_movie_id][after_movie_id] += 1
                if not is_direct:
                    pair_movies[after_movie_id][front_movie_id] += 1
    return pair_movies


def node2vec(num_walks, walk_length, p, q, rating_max_gap_second, is_direct: bool = False):
    random_walks = []
    pairs = load_pair_movies_graph(rating_max_gap_second, is_direct)
    for _ in range(num_walks):
        for node_v in pairs.keys():
            walks = [node_v]
            for _ in range(walk_length):
                if len(walks) == 1:
                    neighbors = pairs[node_v]
                elif node_v in pairs:
                    node_t = walks[-2]
                    zero_node = {node_t: pairs[node_v][node_t] / p} if node_t in pairs[node_v] else {}
                    one_nodes = {node: weight for node, weight in pairs[node_t].items() if node in pairs[node_v]}
                    two_nodes = {node: weight / q for node, weight in pairs[node_v].items()
                                 if node not in zero_node and node not in one_nodes}
                    neighbors = {**zero_node, **one_nodes, **two_nodes}
                else:
                    break
                value_sum = sum(neighbors.values())
                sorted_neighbors = sorted([(key, value / value_sum) for key, value in neighbors.items()],
                                          key=lambda item: item[1], reverse=True)
                walks.append(alias_sample(sorted_neighbors))
                node_v = walks[-1]
            random_walks.append(walks)
    return random_walks


def generate_train_data(output_file, window_size, num_walks, walk_length, p, q, rating_max_gap_second,
                        is_direct: bool = False):
    random_walks = node2vec(num_walks, walk_length, p, q, rating_max_gap_second, is_direct)
    movie_data = load_movie_data()
    records = []
    for walk in random_walks:
        for i in range(window_size, len(walk) - window_size):
            neighbors = []
            for j in range(window_size):
                if i - j - 1 >= 0:
                    neighbors.append(walk[i - j - 1])
                if i + j + 1 < len(walk):
                    neighbors.append(walk[i + j + 1])
            feature = {
                "movie_id": tf.train.Feature(int64_list=tf.train.Int64List(value=[walk[i]]))
            }
            if walk[i] in movie_data:
                if "keywords" in movie_data[walk[i]]:
                    feature["keywords"] = tf.train.Feature(bytes_list=tf.train.BytesList(
                        value=list(map(lambda x: x.encode("utf-8"), movie_data[walk[i]]["keywords"]))))
                if "publishYear" in movie_data[walk[i]]:
                    feature["publishYear"] = tf.train.Feature(int64_list=tf.train.Int64List(
                        value=[movie_data[walk[i]]["publishYear"]]))
                if "categories" in movie_data[walk[i]]:
                    feature["categories"] = tf.train.Feature(bytes_list=tf.train.BytesList(
                        value=list(map(lambda x: x.encode("utf-8"), movie_data[walk[i]]["categories"]))))
            for neighbor in neighbors:
                feature["target_movie_id"] = tf.train.Feature(int64_list=tf.train.Int64List(value=[neighbor]))
                example = tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()
                records.append(example)
    random.shuffle(records)
    with tf.io.TFRecordWriter(output_file) as f:
        for record in records:
            f.write(record)


def input_fn(batch_size, shuffle_buffer_size=None):
    example_schema = {
        "target_movie_id": tf.io.FixedLenFeature(shape=(1,), dtype=tf.int64),
        "movie_id": tf.io.FixedLenFeature(shape=(1,), dtype=tf.int64),
        "keywords": tf.io.RaggedFeature(dtype=tf.string, row_splits_dtype=tf.int64),
        "publishYear": tf.io.FixedLenFeature(shape=(1,), dtype=tf.int64, default_value=0),
        "categories": tf.io.RaggedFeature(dtype=tf.string, row_splits_dtype=tf.int64),
        "label": tf.io.FixedLenFeature(shape=(1,), dtype=tf.int64, default_value=0)
    }
    file_pattern = os.path.abspath(__file__).replace("data/movieLens_graph.py", "data/movieLens/ml-1m/graph.tfrecord")
    return tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=example_schema,
        reader=tf.data.TFRecordDataset,
        label_key="label",
        num_epochs=1,
        shuffle=bool(shuffle_buffer_size),
        shuffle_buffer_size=shuffle_buffer_size,
        drop_final_batch=bool(shuffle_buffer_size))


if __name__ == "__main__":
    train_file = os.path.abspath(__file__).replace("data/movieLens_graph.py", "data/movieLens/ml-1m/graph.tfrecord")
    generate_train_data(train_file, window_size=2, num_walks=5, walk_length=10, p=1, q=1,
                        rating_max_gap_second=999999999999, is_direct=False)

import tensorflow as tf
from tensorflow.python.data.ops.readers import ParallelInterleaveDataset
import os
from collections import defaultdict
import random


user_occupation = {
    0: "other",
    1: "academic/educator",
    2: "artist",
    3: "clerical/admin",
    4: "college/grad student",
    5: "customer service",
    6: "doctor/health care",
    7: "executive/managerial",
    8: "farmer",
    9: "homemaker",
    10: "K-12 student",
    11: "lawyer",
    12: "programmer",
    13: "retired",
    14: "sales/marketing",
    15: "scientist",
    16: "self-employed",
    17: "technician/engineer",
    18: "tradesman/craftsman",
    19: "unemployed",
    20: "writer"
}


def load_movie_data():
    movies = defaultdict(dict)
    path = os.path.abspath(__file__).replace("data/movieLens.py", "data/movieLens/ml-1m/movies.dat")
    with open(path, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            id, name, categories = line.strip("\n").split("::")
            movie_id = int(id)
            publish_year = int(name[name.rindex("(") + 1:-1])
            keywords = name[:name.rindex("(")].replace("(", "").replace(")", "").strip(" ").split(" ")
            categories = categories.split("|")
            if len(keywords) > 0:
                movies[movie_id]["keywords"] = keywords
            if publish_year > 0:
                movies[movie_id]["publishYear"] = publish_year
            if len(categories) > 0:
                movies[movie_id]["categories"] = categories
    return movies


def load_user_data():
    users = defaultdict(dict)
    path = os.path.abspath(__file__).replace("data/movieLens.py", "data/movieLens/ml-1m/users.dat")
    with open(path, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            id, gender, age, occupation, zip_code = line.strip("\n").split("::")
            user_id = int(id)
            users[user_id]["gender"] = gender
            users[user_id]["age"] = int(age)
            users[user_id]["occupation"] = user_occupation[int(occupation)]
            users[user_id]["zip_code"] = zip_code
    return users


def load_rating_data(revert=False):
    ratings = defaultdict(lambda: defaultdict(dict))
    path = os.path.abspath(__file__).replace("data/movieLens.py", "data/movieLens/ml-1m/ratings.dat")
    with open(path, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            user_id, movie_id, rating, timestamp = line.strip("\n").split("::")
            user_id = int(user_id)
            movie_id = int(movie_id)
            if revert:
                ratings[movie_id][user_id]["rating"] = int(rating)
                ratings[movie_id][user_id]["timestamp"] = int(timestamp)
            else:
                ratings[user_id][movie_id]["rating"] = int(rating)
                ratings[user_id][movie_id]["timestamp"] = int(timestamp)
    return ratings


def gen_records(train_output, test_output, train_rate=0.8):
    users = load_user_data()
    movies = load_movie_data()
    ratings = load_rating_data()
    train_records = []
    test_f = tf.io.TFRecordWriter(test_output)
    for user_id in ratings:
        user_behaviors = sorted(list(ratings[user_id].items()), key=lambda item: item[1]["timestamp"])
        user_history_data = {
            "user_history_high_score_movies": [],
            "user_history_low_score_movies": [],
            "user_history_high_score_movie_categories": [],
            "user_history_low_score_movie_categories": [],
            "user_history_high_score_movie_keywords": [],
            "user_history_low_score_movie_keywords": [],
        }
        train_behavior_count = int((len(user_behaviors) + 1) * train_rate)
        behavior_count = 0
        for movie_id, rating_info in user_behaviors:
            rating, timestamp = rating_info["rating"], rating_info["timestamp"]
            example = gen_example(user_id, users[user_id], movie_id, movies[movie_id], rating, user_history_data,
                                  timestamp)
            if behavior_count <= train_behavior_count:
                train_records.append(example.SerializeToString())
            else:
                test_f.write(example.SerializeToString())
            behavior_count += 1
            if rating >= 3:
                user_history_data["user_history_high_score_movies"].append(movie_id)
                if "categories" in movies[movie_id]:
                    user_history_data["user_history_high_score_movie_categories"].append(movies[movie_id]["categories"])
                if "keywords" in movies[movie_id]:
                    user_history_data["user_history_high_score_movie_keywords"].append(movies[movie_id]["keywords"])
            else:
                user_history_data["user_history_low_score_movies"].append(movie_id)
                if "categories" in movies[movie_id]:
                    user_history_data["user_history_low_score_movie_categories"].append(movies[movie_id]["categories"])
                if "keywords" in movies[movie_id]:
                    user_history_data["user_history_low_score_movie_keywords"].append(movies[movie_id]["keywords"])
    test_f.close()
    with tf.io.TFRecordWriter(train_output) as f:
        random.shuffle(train_records)
        for record in train_records:
            f.write(record)


def gen_example(user_id, user_data, movie_id, movie_data, rating, user_history, timestamp):
    context_features = {
        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[1 if rating >= 3 else 0])),
        "rating": tf.train.Feature(float_list=tf.train.FloatList(value=[rating])),
        "timestamp": tf.train.Feature(int64_list=tf.train.Int64List(value=[timestamp])),
        "user_id": tf.train.Feature(int64_list=tf.train.Int64List(value=[user_id])),
        "movie_id": tf.train.Feature(int64_list=tf.train.Int64List(value=[movie_id])),
    }
    sequence_features = {}
    if "gender" in user_data:
        context_features["gender"] = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[user_data["gender"].encode("utf-8")]))
    if "age" in user_data:
        context_features["age"] = tf.train.Feature(int64_list=tf.train.Int64List(value=[user_data["age"]]))
    if "occupation" in user_data:
        context_features["occupation"] = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[user_data["occupation"].encode("utf-8")]))
    if "zip_code" in user_data:
        context_features["zip_code"] = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[user_data["zip_code"].encode("utf-8")]))
    if "keywords" in movie_data:
        context_features["keywords"] = tf.train.Feature(bytes_list=tf.train.BytesList(
            value=list(map(lambda x: x.encode("utf-8"), movie_data["keywords"]))))
    if "publishYear" in movie_data:
        context_features["publishYear"] = tf.train.Feature(int64_list=tf.train.Int64List(value=[movie_data["publishYear"]]))
    if "categories" in movie_data:
        context_features["categories"] = tf.train.Feature(bytes_list=tf.train.BytesList(
            value=list(map(lambda x: x.encode("utf-8"), movie_data["categories"]))))
    if "user_history_high_score_movies" in user_history:
        context_features["user_history_high_score_movies"] = tf.train.Feature(int64_list=tf.train.Int64List(
            value=user_history["user_history_high_score_movies"]))
    if "user_history_low_score_movies" in user_history:
        context_features["user_history_low_score_movies"] = tf.train.Feature(int64_list=tf.train.Int64List(
            value=user_history["user_history_low_score_movies"]))
    if "user_history_high_score_movie_categories" in user_history:
        sequence_features["user_history_high_score_movie_categories"] = tf.train.FeatureList(feature=[
            tf.train.Feature(bytes_list=tf.train.BytesList(
                value=list(map(lambda x: x.encode("utf-8"), movie_categories))))
            for movie_categories in user_history["user_history_high_score_movie_categories"]
        ])
    if "user_history_low_score_movie_categories" in user_history:
        sequence_features["user_history_low_score_movie_categories"] = tf.train.FeatureList(feature=[
            tf.train.Feature(bytes_list=tf.train.BytesList(
                value=list(map(lambda x: x.encode("utf-8"), movie_categories))))
            for movie_categories in user_history["user_history_low_score_movie_categories"]
        ])
    if "user_history_high_score_movie_keywords" in user_history:
        sequence_features["user_history_high_score_movie_keywords"] = tf.train.FeatureList(feature=[
            tf.train.Feature(bytes_list=tf.train.BytesList(
                value=list(map(lambda x: x.encode("utf-8"), movie_keywords))))
            for movie_keywords in user_history["user_history_high_score_movie_keywords"]
        ])
    if "user_history_low_score_movie_keywords" in user_history:
        sequence_features["user_history_low_score_movie_keywords"] = tf.train.FeatureList(feature=[
            tf.train.Feature(bytes_list=tf.train.BytesList(
                value=list(map(lambda x: x.encode("utf-8"), movie_keywords))))
            for movie_keywords in user_history["user_history_low_score_movie_keywords"]
        ])

    return tf.train.SequenceExample(context=tf.train.Features(feature=context_features),
                                    feature_lists=tf.train.FeatureLists(feature_list=sequence_features))


def input_fn(file_pattern, batch_size, num_epochs, label_key, shuffle_buffer_size=None,
             reader_num_threads=tf.data.AUTOTUNE, sloppy_ordering=False):
    example_schema = {
        "label": tf.io.FixedLenFeature(shape=(1,), dtype=tf.int64, default_value=0),
        "rating": tf.io.FixedLenFeature(shape=(1,), dtype=tf.float32, default_value=0),
        "timestamp": tf.io.FixedLenFeature(shape=(1,), dtype=tf.int64, default_value=0),
        "user_id": tf.io.FixedLenFeature(shape=(1,), dtype=tf.int64, default_value=-1),
        "movie_id": tf.io.FixedLenFeature(shape=(1,), dtype=tf.int64, default_value=-1),
        "gender": tf.io.FixedLenFeature(shape=(1,), dtype=tf.string, default_value=""),
        "age": tf.io.FixedLenFeature(shape=(1,), dtype=tf.int64, default_value=0),
        "occupation": tf.io.FixedLenFeature(shape=(1,), dtype=tf.string, default_value=""),
        "zip_code": tf.io.FixedLenFeature(shape=(1,), dtype=tf.string, default_value=""),
        "keywords": tf.io.RaggedFeature(dtype=tf.string, row_splits_dtype=tf.int64),
        "publishYear": tf.io.FixedLenFeature(shape=(1,), dtype=tf.int64, default_value=0),
        "categories": tf.io.RaggedFeature(dtype=tf.string, row_splits_dtype=tf.int64),
        "user_history_high_score_movies": tf.io.RaggedFeature(dtype=tf.int64, row_splits_dtype=tf.int64),
        "user_history_low_score_movies": tf.io.RaggedFeature(dtype=tf.int64, row_splits_dtype=tf.int64),
        "user_history_high_score_movie_categories": tf.io.RaggedFeature(dtype=tf.string, row_splits_dtype=tf.int64),
        "user_history_low_score_movie_categories": tf.io.RaggedFeature(dtype=tf.string, row_splits_dtype=tf.int64),
        "user_history_high_score_movie_keywords": tf.io.RaggedFeature(dtype=tf.string, row_splits_dtype=tf.int64),
        "user_history_low_score_movie_keywords": tf.io.RaggedFeature(dtype=tf.string, row_splits_dtype=tf.int64),
    }
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
    def apply_parse_fn(dataset):
        pass
    dataset = dataset.apply()
    if label_key is not None:
        if label_key not in example_schema:
            raise ValueError("The 'label_key' provided (%r) must be one of the 'features' keys." % label_key)
        dataset = dataset.map(lambda x: (x, x.pop(label_key)))
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def train_input_fn(batch_size, label_key="rating"):
    path = os.path.abspath(__file__).replace("data/movieLens.py", "data/movieLens/ml-1m/train.tfrecord")
    return input_fn(path, batch_size, num_epochs=1, label_key=label_key, shuffle_buffer_size=batch_size * 10)


def test_input_fn(batch_size, label_key="rating"):
    path = os.path.abspath(__file__).replace("data/movieLens.py", "data/movieLens/ml-1m/test.tfrecord")
    return input_fn(path, batch_size, num_epochs=1, label_key=label_key, shuffle_buffer_size=None)


"""
    timestamp = tf.keras.layers.Input(shape=(1,), name="timestamp", dtype=tf.int64)
    user_id = tf.keras.layers.Input(shape=(1,), name="user_id", dtype=tf.int64)
    movie_id = tf.keras.layers.Input(shape=(1,), name="movie_id", dtype=tf.int64)
    gender = tf.keras.layers.Input(shape=(1,), name="gender", dtype=tf.string)
    age = tf.keras.layers.Input(shape=(1,), name="age", dtype=tf.int64)
    occupation = tf.keras.layers.Input(shape=(1,), name="occupation", dtype=tf.string)
    zip_code = tf.keras.layers.Input(shape=(1,), name="zip_code", dtype=tf.string)
    keywords = tf.keras.layers.Input(shape=(None,), name="keywords", dtype=tf.string, ragged=True)
    publish_year = tf.keras.layers.Input(shape=(1,), name="publishYear", dtype=tf.int64)
    categories = tf.keras.layers.Input(shape=(None,), name="categories", dtype=tf.string, ragged=True)
    user_history_high_score_movies = tf.keras.layers.Input(
        shape=(None,), name="user_history_high_score_movies", dtype=tf.int64, ragged=True)
    user_history_low_score_movies = tf.keras.layers.Input(
        shape=(None,), name="user_history_low_score_movies", dtype=tf.int64, ragged=True)
    user_history_high_score_movie_categories = tf.keras.layers.Input(
        shape=(None,), name="user_history_high_score_movie_categories", dtype=tf.string, ragged=True)
    user_history_low_score_movie_categories = tf.keras.layers.Input(
        shape=(None,), name="user_history_low_score_movie_categories", dtype=tf.string, ragged=True)
    user_history_high_score_movie_keywords = tf.keras.layers.Input(
        shape=(None,), name="user_history_high_score_movie_keywords", dtype=tf.string, ragged=True)
    user_history_low_score_movie_keywords = tf.keras.layers.Input(
        shape=(None,), name="user_history_low_score_movie_keywords", dtype=tf.string, ragged=True)
"""


if __name__ == "__main__":
    train_file = os.path.abspath(__file__).replace("data/movieLens.py", "data/movieLens/ml-1m/train2.tfrecord")
    test_file = os.path.abspath(__file__).replace("data/movieLens.py", "data/movieLens/ml-1m/test2.tfrecord")
    gen_records(train_file, test_file, train_rate=0.8)

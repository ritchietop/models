import tensorflow as tf
import os
from collections import defaultdict


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


def load_rating_data():
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
            ratings[user_id][movie_id]["rating"] = int(rating)
            ratings[user_id][movie_id]["timestamp"] = int(timestamp)
    return ratings


def gen_records(train_output, test_output, train_rate=0.8):
    users = load_user_data()
    movies = load_movie_data()
    ratings = load_rating_data()
    train_file = tf.io.TFRecordWriter(train_output)
    test_file = tf.io.TFRecordWriter(test_output)
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
            example = gen_example(users[user_id], movie_id, movies[movie_id], rating, user_history_data, timestamp)
            if behavior_count <= train_behavior_count:
                train_file.write(example.SerializeToString())
            else:
                test_file.write(example.SerializeToString())
            behavior_count += 1
            if rating >= 3:
                user_history_data["user_history_high_score_movies"].append(movie_id)
                if "categories" in movies[movie_id]:
                    for category in movies[movie_id]["categories"]:
                        user_history_data["user_history_high_score_movie_categories"].append(category)
                if "keywords" in movies[movie_id]:
                    for keyword in movies[movie_id]["keywords"]:
                        user_history_data["user_history_high_score_movie_keywords"].append(keyword)
            else:
                user_history_data["user_history_low_score_movies"].append(movie_id)
                if "categories" in movies[movie_id]:
                    for category in movies[movie_id]["categories"]:
                        user_history_data["user_history_low_score_movie_categories"].append(category)
                if "keywords" in movies[movie_id]:
                    for keyword in movies[movie_id]["keywords"]:
                        user_history_data["user_history_low_score_movie_keywords"].append(keyword)
    train_file.close()
    test_file.close()


def gen_example(user_data, movie_id, movie_data, rating, user_history, timestamp):
    features = {
        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[1 if rating >= 3 else 0])),
        "rating": tf.train.Feature(float_list=tf.train.FloatList(value=[rating])),
        "timestamp": tf.train.Feature(int64_list=tf.train.Int64List(value=[timestamp])),
        "movie_id": tf.train.Feature(int64_list=tf.train.Int64List(value=[movie_id])),
    }
    if "gender" in user_data:
        features["gender"] = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[user_data["gender"].encode("utf-8")]))
    if "age" in user_data:
        features["age"] = tf.train.Feature(int64_list=tf.train.Int64List(value=[user_data["age"]]))
    if "occupation" in user_data:
        features["occupation"] = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[user_data["occupation"].encode("utf-8")]))
    if "zip_code" in user_data:
        features["zip_code"] = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[user_data["zip_code"].encode("utf-8")]))
    if "keywords" in movie_data:
        features["keywords"] = tf.train.Feature(bytes_list=tf.train.BytesList(
            value=list(map(lambda x: x.encode("utf-8"), movie_data["keywords"]))))
    if "publishYear" in movie_data:
        features["publishYear"] = tf.train.Feature(int64_list=tf.train.Int64List(value=[movie_data["publishYear"]]))
    if "categories" in movie_data:
        features["categories"] = tf.train.Feature(bytes_list=tf.train.BytesList(
            value=list(map(lambda x: x.encode("utf-8"), movie_data["categories"]))))
    if "user_history_high_score_movies" in user_history:
        features["user_history_high_score_movies"] = tf.train.Feature(int64_list=tf.train.Int64List(
            value=user_history["user_history_high_score_movies"]))
    if "user_history_low_score_movies" in user_history:
        features["user_history_low_score_movies"] = tf.train.Feature(int64_list=tf.train.Int64List(
            value=user_history["user_history_low_score_movies"]))
    if "user_history_high_score_movie_categories" in user_history:
        features["user_history_high_score_movie_categories"] = tf.train.Feature(bytes_list=tf.train.BytesList(
            value=list(map(lambda x: x.encode("utf-8"), user_history["user_history_high_score_movie_categories"]))))
    if "user_history_low_score_movie_categories" in user_history:
        features["user_history_low_score_movie_categories"] = tf.train.Feature(bytes_list=tf.train.BytesList(
            value=list(map(lambda x: x.encode("utf-8"), user_history["user_history_low_score_movie_categories"]))))
    if "user_history_high_score_movie_keywords" in user_history:
        features["user_history_high_score_movie_keywords"] = tf.train.Feature(bytes_list=tf.train.BytesList(
            value=list(map(lambda x: x.encode("utf-8"), user_history["user_history_high_score_movie_keywords"]))))
    if "user_history_low_score_movie_keywords" in user_history:
        features["user_history_low_score_movie_keywords"] = tf.train.Feature(bytes_list=tf.train.BytesList(
            value=list(map(lambda x: x.encode("utf-8"), user_history["user_history_low_score_movie_keywords"]))))

    return tf.train.Example(features=tf.train.Features(feature=features))


def input_fn(file_pattern, batch_size, num_epochs, shuffle_buffer_size=None):
    example_schema = {
        "label": tf.io.FixedLenFeature(shape=(1,), dtype=tf.int64, default_value=0),
        "rating": tf.io.FixedLenFeature(shape=(1,), dtype=tf.float32, default_value=0),
        "timestamp": tf.io.FixedLenFeature(shape=(1,), dtype=tf.int64, default_value=0),
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
    return tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=example_schema,
        reader=tf.data.TFRecordDataset,
        label_key="rating",
        num_epochs=num_epochs,
        shuffle=bool(shuffle_buffer_size),
        shuffle_buffer_size=shuffle_buffer_size,
        drop_final_batch=bool(shuffle_buffer_size))


def train_input_fn(batch_size):
    path = os.path.abspath(__file__).replace("data/movieLens.py", "data/movieLens/ml-1m/train.tfrecord")
    return input_fn(path, batch_size, num_epochs=1, shuffle_buffer_size=batch_size * 10)


def test_input_fn(batch_size):
    path = os.path.abspath(__file__).replace("data/movieLens.py", "data/movieLens/ml-1m/test.tfrecord")
    return input_fn(path, batch_size, num_epochs=1, shuffle_buffer_size=None)


"""
    timestamp = tf.keras.layers.Input(shape=(1,), name="timestamp", dtype=tf.int64)
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
    train_file = os.path.abspath(__file__).replace("data/movieLens.py", "data/movieLens/ml-1m/train.tfrecord")
    test_file = os.path.abspath(__file__).replace("data/movieLens.py", "data/movieLens/ml-1m/test.tfrecord")
    gen_records(train_file, test_file, train_rate=0.8)

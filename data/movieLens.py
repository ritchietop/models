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
            publish_year = int(name[name.rindex("(") + 1:-1])
            keywords = name[:name.rindex("(")].replace("(", "").replace(")", "").strip(" ").split(" ")
            categories = categories.split("|")
            if len(keywords) > 0:
                movies[id]["keywords"] = keywords
            if publish_year > 0:
                movies[id]["publishYear"] = publish_year
            if len(categories) > 0:
                movies[id]["categories"] = categories
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
            users[id]["gender"] = gender
            users[id]["age"] = int(age)
            users[id]["occupation"] = user_occupation[int(occupation)]
            users[id]["zip_code"] = zip_code
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
            ratings[user_id][movie_id]["rating"] = int(rating)
            ratings[user_id][movie_id]["timestamp"] = int(timestamp)
    return ratings


def gen_records(output):
    users = load_user_data()
    movies = load_movie_data()
    ratings = load_rating_data()
    with open(output, "w") as f:
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
            for movie_id, rating_info in user_behaviors:
                rating, timestamp = rating_info["rating"], rating_info["timestamp"]
                example = gen_example(users[user_id], movie_id, movies[movie_id], rating, user_history_data, timestamp)
                f.write(example.SerializeToString())
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


def gen_example(user_data, movie_id, movie_data, rating, user_history_data, timestamp):
    features = {
        "rating": tf.train.Feature(float_list=tf.train.FloatList(value=[rating])),
        "timestamp": tf.train.Feature(int64_list=tf.train.Int64List(value=[timestamp])),
        "movie_id": tf.train.Feature(int64_list=tf.train.Int64List(value=[movie_id])),
    }

    return tf.train.Example(features=tf.train.Features(feature=features))


if __name__ == "__main__":
    output = path = os.path.abspath(__file__).replace("data/movieLens.py", "data/movieLens/ml-1m/train.tfrecord")
    gen_records(output)

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
            publish_year = int(name[name.rindex("(")+1:-1])
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


def gen_record(user_data, movie_data, rating, user_history_data):
    pass


if __name__ == "__main__":
    users = load_user_data()
    movies = load_movie_data()
    ratings = load_rating_data()
    for user_id in ratings:
        for movie_id in ratings[user_id]:
            rating = ratings[user_id][movie_id]


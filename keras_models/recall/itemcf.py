from data.movieLens import load_rating_data


rating_data = load_rating_data(revert=True)
for movie_id_a in rating_data:
    for movie_id_b in rating_data:
        if movie_id_a != movie_id_b:


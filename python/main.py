import torch
from fastapi import FastAPI
import predict as p
import pandas as pd
import models as m

app = FastAPI()

movie_path = "created_models/movie.csv"
movie_data = pd.read_csv(movie_path)
data_path = "created_models/user_movies.csv"
user_movie_data = pd.read_csv(data_path)
num_users = user_movie_data['userId'].nunique()
num_movies = user_movie_data['movieId'].nunique()

@app.get("/")
def read_root():
    print(f"Number of users: {num_users}, Number of movies: {num_movies}")
    return "Welcome to the Movie Recommendation API!"

@app.get("/api/predict/collabrative/one-genre/{genre}")
def recommend_collaborative_one_genre(genre: str):
    model_path = "created_models/cosine_similarity_model.pth"
    model = m.CosineSimilarityModel(num_users, num_movies, 19)
    model.load_state_dict(torch.load(model_path))
    predictions = p.recommend_moviesbyOneGenre(genre, model, user_movie_data)
    for movie_id in predictions:
        movie_title, genre = movie_data[movie_data['movieId'] == movie_id][['title', 'genres']].values[0]
        print(f"Movie ID: {movie_id}, Movie Title: {movie_title}, Genre: {genre}")
    return predictions

@app.get("/api/predict/ /multi-genre/{genre}")
def recommend_collaborative_multi_genre(genre: str):
    model_path = "created_models/cosine_similarity_model.pth"
    model = m.CosineSimilarityModel(num_users, num_movies, 19)
    model.load_state_dict(torch.load(model_path))
    predictions = p.recommend_moviesbyMultiGenre(genre, model, user_movie_data)
    for movie_id in predictions:
        movie_title, genre = movie_data[movie_data['movieId'] == movie_id][['title', 'genres']].values[0]
        print(f"Movie ID: {movie_id}, Movie Title: {movie_title}, Genre: {genre}")
    return predictions

@app.get("/api/predict/content/one-genre/{genre}")
def recommend_content_one_genre(genre: str):
    model_path = "created_models/matrix_factorization_model.pth"
    model = m.MatrixFactorizationModel(num_users, num_movies, 50)
    model.load_state_dict(torch.load(model_path))
    predictions = p.recommend_moviesbyOneGenre(genre, model, user_movie_data)
    for movie_id in predictions:
        movie_title, genre = movie_data[movie_data['movieId'] == movie_id][['title', 'genres']].values[0]
        print(f"Movie ID: {movie_id}, Movie Title: {movie_title}, Genre: {genre}")
    return predictions

@app.get("/api/predict/content/multi-genre/{genre}")
def recommend_content_multi_genre(genre: str):
    model_path = "created_models/matrix_factorization_model.pth"
    model = m.MatrixFactorizationModel(num_users, num_movies, 50)
    model.load_state_dict(torch.load(model_path))
    predictions = p.recommend_moviesbyMultiGenre(genre, model, user_movie_data)
    for movie_id in predictions:
        movie_title, genre = movie_data[movie_data['movieId'] == movie_id][['title', 'genres']].values[0]
        print(f"Movie ID: {movie_id}, Movie Title: {movie_title}, Genre: {genre}")
    return predictions

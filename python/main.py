from http.client import HTTPException
import torch
from fastapi import FastAPI
import predict as p
import pandas as pd
import models as m
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np


app = FastAPI()

# Load the datasets
movies_df = pd.read_csv("created_models/movie.csv")
user_movies_df = pd.read_csv("created_models/user_movies.csv")

num_users = user_movies_df['userId'].nunique()
num_movies = user_movies_df['movieId'].nunique()
user_movie_matrix = user_movies_df.pivot(index='userId', columns='movieId', values='rating').fillna(0).to_numpy()

def preprocess_genres(movies_df):
    mlb = MultiLabelBinarizer()
    genre_data = movies_df['genres'].str.split('|')
    genre_matrix = mlb.fit_transform(genre_data)
    return genre_matrix, mlb

genre_matrix, genre_encoder = preprocess_genres(movies_df)
model_path = "created_models/autoencoder_model.pth"
loaded_autoencoder = m.GenreAutoencoder(20, 16)  # Assuming you defined GenreAutoencoder earlier
loaded_autoencoder.load_state_dict(torch.load(model_path))
loaded_autoencoder.eval()

movies_df = pd.read_csv("created_models/movie.csv")  # Make sure movie.csv is in the correct path
data = genre_matrix # Make sure genre_matrix is defined and available

embedding_dim = 16  # Make sure this matches the embedding_dim used during training
loaded_ncf = m.NCF(num_users, num_movies, embedding_dim)
loaded_ncf.load_state_dict(torch.load('created_models/ncf_model.pth'))
loaded_ncf.eval()

data_tensor = torch.FloatTensor(data)
with torch.no_grad():
    embeddings, _ = loaded_autoencoder(data_tensor)
embeddings = embeddings.numpy()
loaded_similarity_matrix = cosine_similarity(embeddings)

def recommend_movies_loaded(movie_ids, top_n=5):
    recommended_movies = []
    for movie_id in movie_ids:
        try:
            movie_idx = movies_df[movies_df['movieId'] == movie_id].index[0]
            similar_movies = loaded_similarity_matrix[movie_idx].argsort()[::-1][1:top_n+1]
            for sim_movie_idx in similar_movies:
                recommended_movies.append(movies_df.iloc[sim_movie_idx])
        except IndexError:
            print(f"Movie with ID {movie_id} not found in the dataset.")
    return pd.DataFrame(recommended_movies)

@app.get("/")
def read_root():
    return "Welcome to the Movie Recommendation API!"

@app.get("/api/predict/collabrative/{user_id}")
def recommend_collaborative_one_genre(user_id: str):
    try:
        user_id = int(user_id)
        user_movies = user_movie_matrix[user_id]
        unwatched_movie_ids = np.where(user_movies == 0)[0]

        with torch.no_grad():
            user_ids_tensor = torch.LongTensor([user_id] * len(unwatched_movie_ids))
            movie_ids_tensor = torch.LongTensor(unwatched_movie_ids)
            predictions = loaded_ncf(user_ids_tensor, movie_ids_tensor).numpy()

        recommended_movie_ids = unwatched_movie_ids[np.argsort(predictions)[::-1][:5]]
        response = []
        for movie_id in recommended_movie_ids:
            movie_title = movies_df.iloc[movie_id]['title']
            movie_genres = movies_df.iloc[movie_id]['genres']
            similarity_score = predictions[np.where(unwatched_movie_ids == movie_id)][0]
            response.append({
                "movieId": int(movie_id),
                "title": movie_title,
                "genres": movie_genres,
            })
        return response

    except IndexError:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.get("/api/predict/content/{movie_ids}")
def recommend_content_multi_genre(movie_ids: str):
    movie_ids_to_recommend = [int(movie_id) for movie_id in movie_ids.split(",")]
    try:
        recommendations = recommend_movies_loaded(movie_ids_to_recommend)
        return recommendations[['movieId', 'title', 'genres']].to_dict(orient='records') # Return as a list of dictionaries
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


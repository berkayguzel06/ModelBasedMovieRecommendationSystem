import numpy as np
import torch

def test_model(model, user_id, movie_ids):
    """
    Tests the model for a given user and a list of movie IDs.

    Args:
      model: The trained model.
      user_id: The ID of the user.
      movie_ids: A list of movie IDs.

    Returns:
      A list of predicted ratings for the given movie IDs, along with the movie titles.
    """

    # Convert the user ID and movie IDs to PyTorch tensors
    user_id_tensor = torch.tensor([user_id], dtype=torch.long)
    movie_ids_tensor = torch.tensor(movie_ids, dtype=torch.long)

    with torch.no_grad():
        # Make predictions for the given user and movie IDs
        predictions = model(user_id_tensor, movie_ids_tensor)

    return predictions.tolist()

def recommend_moviesbyMultiGenre(genres, model, movies, top_n=5):
    """
    Recommends movies based on multiple genres.

    Args:
        genres: A list of genres.
        model: The trained model to use for recommendations.
        top_n: The number of top recommendations to return (default: 5).

    Returns:
        A list of recommended movie titles.
    """

    # Filter movies by any of the specified genres
    genre_movies = movies[movies[genres].any(axis=1)]

    if genre_movies.empty:
        return ["No movies found for these genres."]

    # Get unique movie IDs for the specified genres
    movie_ids = genre_movies['movieId'].unique().tolist()

    # Select a random user for demonstration
    random_user_id = np.random.choice(movies['userId'].unique())

    # Get predictions for the random user and all movies of that genre
    predictions = test_model(model, random_user_id, movie_ids)

    # Ensure predictions is a list of ratings and match with movie IDs
    movie_recommendations = list(zip(movie_ids, predictions))

    # Sort movies by predicted ratings in descending order
    movie_recommendations.sort(key=lambda x: x[1], reverse=True)

    # Return the top_n movie IDs
    top_movie_ids = [movie_id for movie_id, _ in movie_recommendations[:top_n]]
    return top_movie_ids


# prompt: genre vererek film önermesini sağlamak istiyourm onun için test fonskiyonu yazarmısın. User_Id ve movie_Id olmadan önermesi laızm

def recommend_moviesbyOneGenre(genre, model, movies, top_n=5):
    """
    Recommends movies based on a given genre.

    Args:
        genre: The genre of movies to recommend.
        model: The trained model to use for recommendations.
        movies: The movies dataset containing movie details and user IDs.
        top_n: The number of top recommendations to return (default: 5).

    Returns:
        A list of recommended movie IDs.
    """

    # Filter movies by genre
    genre_movies = movies[movies[genre] == 1]

    if genre_movies.empty:
        return ["No movies found for this genre."]

    # Get unique movie IDs for the specified genre
    movie_ids = genre_movies['movieId'].unique().tolist()

    # Select a random user for demonstration
    random_user_id = np.random.choice(movies['userId'].unique())
    
    # Get predictions for the random user and all movies of that genre
    predictions = test_model(model, random_user_id, movie_ids)

    # Ensure predictions is a list of ratings and match with movie IDs
    movie_recommendations = list(zip(movie_ids, predictions))

    # Sort movies by predicted ratings in descending order
    movie_recommendations.sort(key=lambda x: x[1], reverse=True)

    # Return the top_n movie IDs
    top_movie_ids = [movie_id for movie_id, _ in movie_recommendations[:top_n]]
    return top_movie_ids


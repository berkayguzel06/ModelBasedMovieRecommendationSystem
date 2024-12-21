import torch.nn as nn
import torch

class CosineSimilarityModel(nn.Module):
    def __init__(self, num_users, num_movies, feature_size):
        super(CosineSimilarityModel, self).__init__()
        self.user_embeddings = nn.Embedding(num_users, feature_size)
        self.movie_embeddings = nn.Embedding(num_movies, feature_size)

    def forward(self, user_id, movie_id):
        user_embedding = self.user_embeddings(user_id)
        movie_embedding = self.movie_embeddings(movie_id)
        similarity = nn.functional.cosine_similarity(user_embedding, movie_embedding, dim=-1)
        return similarity
    
class MatrixFactorizationModel(nn.Module):
    def __init__(self, num_users, num_movies, embedding_size):
        super(MatrixFactorizationModel, self).__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_size)
        self.movie_embeddings = nn.Embedding(num_movies, embedding_size)

    def forward(self, user_id, movie_id):
        user_embedding = self.user_embeddings(user_id)
        movie_embedding = self.movie_embeddings(movie_id)
        # Dot product for interaction
        interaction = torch.sum(user_embedding * movie_embedding, dim=1)
        return interaction
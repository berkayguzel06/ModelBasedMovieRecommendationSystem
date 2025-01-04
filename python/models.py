import torch.nn as nn
import torch
from sklearn.metrics.pairwise import cosine_similarity

class GenreAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GenreAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class NCF(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim):
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, user_ids, movie_ids):
        user_embeds = self.user_embedding(user_ids)
        movie_embeds = self.movie_embedding(movie_ids)
        x = torch.cat([user_embeds, movie_embeds], dim=-1)
        return self.fc_layers(x).squeeze()


class GenreBasedRecommender:
    def __init__(self, similarity_matrix, genre_vectors, mlb_classes, titles):
        self.similarity_matrix = similarity_matrix
        self.genre_vectors = genre_vectors
        self.mlb_classes = mlb_classes
        self.titles = titles
    
    def recommend_by_genre(self, genre_list, top_n=5):
        # Genre listesini vektöre çevir ve GPU'ya taşı
        genre_vector = torch.tensor(
            [1 if genre in genre_list else 0 for genre in self.mlb_classes]
        )
        
        # Genre vektörü ile filmlerin benzerlik skorlarını hesapla (GPU'da)
        similarity_scores = cosine_similarity(
            genre_vector.cpu().numpy().reshape(1, -1),
            self.genre_vectors.cpu().numpy()
        )[0]
        
        # Benzerlik skorlarına göre sırala
        top_movie_indices = similarity_scores.argsort()[::-1][:top_n]
        
        # Önerilen filmleri yazdır
        return [self.titles[i] for i in top_movie_indices]

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
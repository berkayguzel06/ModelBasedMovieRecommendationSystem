from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
from sqlalchemy import create_engine

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:1234@localhost:5432/warehouse'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

db_config = {
    'user': 'postgres',
    'password': '1234',
    'host': 'localhost',
    'port': '5432',
    'database': 'warehouse'
}

connection_string = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
# CORS Policy for http://localhost:3000
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response

@app.route('/api/movies/random-movies', methods=['GET'])
def fetch_random_movies():
    
    # Rastgele 6 film seçmek için sorgu
    query = "SELECT * FROM movies ORDER BY RANDOM() LIMIT 6;"


# SQLAlchemy motoru oluşturma
    engine = create_engine(connection_string)
    movies = pd.read_sql_query(query, engine)

    # Rastgele seçilen filmleri yazdır
    print(movies)
    movies_list = movies.to_dict(orient='records')
    

    return jsonify(movies_list)

@app.route('/api/movies/all-movies', methods=['GET'])
def fetch_all_movies():
    
    # Rastgele 6 film seçmek için sorgu
    query = "SELECT * FROM movies;"


# SQLAlchemy motoru oluşturma
    engine = create_engine(connection_string)
    movies = pd.read_sql_query(query, engine)

    # Rastgele seçilen filmleri yazdır
    print(movies)
    movies_list = movies.to_dict(orient='records')
    

    return jsonify(movies_list)


@app.route('/api/movies/<int:movie_id>', methods=['GET'])
def fetch_movie_by_id(movie_id):
    # Belirli bir movie_id'ye göre filmi seçmek için sorgu
    query = f"SELECT * FROM movies WHERE movieId = {movie_id};"

    # SQLAlchemy motoru oluşturma
    engine = create_engine(connection_string)
    movie = pd.read_sql_query(query, engine)

    # Eğer film bulunamazsa
    if movie.empty:
        return jsonify({'error': 'Film bulunamadı'}), 404

    # DataFrame'i JSON'a dönüştürme
    movie_dict = movie.to_dict(orient='records')[0]

    return jsonify(movie_dict)



# Initialize the database
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True)

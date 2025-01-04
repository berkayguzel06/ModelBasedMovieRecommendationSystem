import pandas as pd
from sqlalchemy import create_engine

# PostgreSQL bağlantı detayları
db_config = {
    'user': 'postgres',
    'password': '1234',
    'host': 'localhost',
    'port': '5432',  # Örneğin: 5432
    'database': 'warehouse'
}

# Bağlantı stringi oluşturma
connection_string = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"

# SQLAlchemy motoru oluşturma
engine = create_engine(connection_string)

# CSV dosyasını DataFrame olarak okuma
df = pd.read_csv('movie.csv')

# DataFrame'i PostgreSQL tablosuna yazma
df.to_sql('movies', engine, if_exists='replace', index=False)

print("CSV dosyası PostgreSQL veritabanına başarıyla yazıldı.")
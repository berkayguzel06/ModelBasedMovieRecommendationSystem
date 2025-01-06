"use client";

import clsx from 'clsx';
import Image from 'next/image';
import { useParams, useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';

export default function MoviePage() {
  const router = useRouter();
  const { movieId } = useParams(); // URL'den movieId'yi al
  const [movie, setMovie] = useState(null);
  const [recommendations, setRecommendations] = useState([]);

  useEffect(() => {
    if (!movieId) return; // movieId yoksa API çağrısı yapma

    // API'den film bilgilerini çekme
    const fetchMovie = async () => {
      try {
        const response = await fetch(`http://127.0.0.1:5000/api/movies/${movieId}`);
        if (!response.ok) {
          throw new Error('Film bilgileri alınamadı');
        }
        const data = await response.json();
        setMovie(data);
      } catch (error) {
        console.error('Film bilgileri alınırken hata oluştu:', error);
      }
    };

    // API'den önerilen filmleri çekme
    const fetchRecommendations = async () => {
      try {
        const response = await fetch(`http://127.0.0.1:8000/api/predict/content/${movieId}`);
        if (!response.ok) {
          throw new Error('Önerilen filmler alınamadı');
        }
        const data = await response.json();
        console.log(data);
        setRecommendations(data);
      } catch (error) {
        console.error('Önerilen filmler alınırken hata oluştu:', error);
      }
    };

    fetchMovie();
    fetchRecommendations();
  }, [movieId]); // movieId değiştiğinde useEffect yeniden çalışır

  if (!movie) {
    return <div>Loading...</div>;
  }

  const MovieHeader = ({ title, year }) => (
    <header className="text-center">
      <h1 className="text-4xl font-bold ">{title}</h1>
      <p className="text-xl text-muted-foreground mt-2 ">{year}</p>
    </header>
  );

  const MovieInfo = ({ movie }) => (
    <div className="space-y-4">
      <div>
        <h2 className="text-4xl font-semibold">Movie Info</h2>
        <p><strong>Title:</strong> {movie.title}</p>
        <p><strong>Genre:</strong> {movie.genres}</p>
      </div>
    </div>
  );

  const MoviePoster = ({ src, alt }) => (
    <div className="relative aspect-[2/3] w-full">
      <Image
        src={src}
        alt={alt}
        fill
        className="object-cover rounded-lg shadow-lg"
      />
    </div>
  );

  const MovieRecommendations = ({ recommendations }) => (
    <div className="mt-12">
      <h2 className="text-2xl font-semibold mb-4 ">Recommended Movies</h2>
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-4">
        {recommendations.map((movie) => (
          <div key={movie.movieId} className="text-center group"
          onClick={() => {
            router.push(`/moviePage/${movie.movieId}`);
          }}>
            <div className="relative aspect-[2/3] w-full mb-2 overflow-hidden rounded-lg shadow-md transition-transform duration-300 ease-in-out group-hover:scale-105">
              <Image
                src="/poster.jpg"
                alt={movie.title}
                fill
                className={clsx(
                  "object-cover transition-transform duration-300 ease-in-out group-hover:scale-110",
                  "transform-gpu"
                )}
              />
            </div>
            <h3 className="font-semibold mt-5">{movie.title}</h3>
            <p className="text-sm text-muted-foreground">{movie.year}</p>
          </div>
        ))}
      </div>
    </div>
  );

  return (
    <div className="px-4 py-8 bg-gradient-animated">
      <div className="rounded-lg shadow-lg p-8 bg-gray-100 bg-opacity-75">
        <MovieHeader title={movie.title} year={movie.year} />
        <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-8">
          <div className="md:col-span-1">
            <MoviePoster src="/poster.jpg" alt={movie.title} />
          </div>
          <div className="md:col-span-2">
            <MovieInfo movie={movie} />
          </div>
        </div>
        <MovieRecommendations recommendations={recommendations} />
      </div>
    </div>
  );
}
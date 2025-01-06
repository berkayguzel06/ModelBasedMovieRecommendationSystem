"use client"
import { ArrowLeft } from 'lucide-react';
import Link from "next/link";
import { useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';

const SurveyPage = () => {
  const router = useRouter();
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [direction, setDirection] = useState(0);
  const [surveyMovies, setSurveyMovies] = useState([]);
  const [selectedMovieIds, setSelectedMovieIds] = useState([]); // Seçilen film ID'leri
  const [recommendedMovies, setRecommendedMovies] = useState([]); // Önerilen filmler

  const fetchSurveyMovies = async () => {
    const response = await fetch('http://127.0.0.1:5000/api/movies/random-movies');
    return await response.json();
  };

  useEffect(() => {
    const fetchMovies = async () => {
      const movies = [];
      for (let i = 0; i < 3; i++) {
        const movieBatch = await fetchSurveyMovies();
        movies.push(...movieBatch);
      }
      setSurveyMovies(movies);
    };

    fetchMovies();
  }, []);

  const surveyQuestions = [
    {
      id: 1,
      question: "Pick a movie!",
      choices: surveyMovies.slice(0, 6) // İlk 6 film
    },
    {
      id: 2,
      question: "Pick another movie!",
      choices: surveyMovies.slice(6, 12) // Sonraki 6 film
    },
    {
      id: 3,
      question: "One last pick!",
      choices: surveyMovies.slice(12, 18) // Son 6 film
    }
  ];

  const paginate = (newDirection) => {
    if (
      (currentQuestionIndex === 0 && newDirection === -1) ||
      (currentQuestionIndex === surveyQuestions.length - 1 && newDirection === 1)
    ) {
      return;
    }
    setDirection(newDirection);
    setCurrentQuestionIndex(currentQuestionIndex + newDirection);
  };

  const handleMovieSelection = (movieId) => {
    setSelectedMovieIds((prevIds) => [...prevIds, movieId]); // Seçilen film ID'sini ekle

    if (currentQuestionIndex === surveyQuestions.length - 1) {
      // Son soruda ise API'ye istek at
      fetchRecommendedMovies(selectedMovieIds);
    } else {
      // Son soru değilse bir sonraki soruya geç
      paginate(1);
    }
  };

  const fetchRecommendedMovies = async (movieIds) => {
    const movieIdsString = movieIds.join(','); // ID'leri virgülle birleştir
    const response = await fetch(`http://127.0.0.1:8000/api/predict/content/${movieIdsString}`);
    const data = await response.json(); //9,65,45
    setRecommendedMovies(data); // Önerilen filmleri state'e kaydet
  };

  return (
    <div className="min-h-screen bg-gradient-animated">
      {/* Navbar */}
      <header className="max-w-7xl mx-auto px-4 py-8">
        <nav className="flex justify-between items-center">
          <button onClick={() => {}} className="text-2xl font-bold text-white">
            <Link href="/">MovieMagic</Link>
          </button>
          <button 
            onClick={() => {}}
            className="text-white hover:bg-white/20 transition-colors duration-200 text-lg font-medium bg-white/10 rounded-full px-6 py-1"
          >
            <Link href="/movieSetPage">Movies</Link>
          </button>
        </nav>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-12 relative">
        {recommendedMovies.length > 0 ? (
          // Önerilen filmler gösteriliyor
          <div>
            <h2 className="text-2xl font-bold text-white mb-8">Önerilen Filmler</h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-8">
              {recommendedMovies.map((movie) => (
                <div 
                  key={movie.movieId} 
                  className="group bg-white/90 backdrop-blur-sm rounded-xl overflow-hidden border border-gray-200 transition-all duration-300 hover:shadow-xl hover:shadow-blue-500/10 cursor-pointer hover:scale-105"
                  onClick={() => {
                    router.push(`/moviePage/${movie.movieId}`); // Film ID'sine göre yönlendirme
                  }}
                >
                  <div className="relative h-48 m-3 rounded-lg bg-gradient-to-br from-gray-100 to-gray-50 flex items-center justify-center">
                  </div>
                  <div className="p-6">
                    <h2 className="text-xl font-bold text-gray-800">
                      {movie.title}
                    </h2>
                  </div>
                  <div className="p-2 ml-4">
                    <p className="text-s text-gray-800">
                      {movie.genres}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ) : (
          // Anket soruları gösteriliyor
          <div className="mb-16">
            <div className="flex items-center mb-8">
              <button 
                onClick={() => paginate(-1)}
                disabled={currentQuestionIndex === 0}
                className={`text-white hover:bg-white/20 transition-colors duration-200 bg-white/10 rounded-full p-2 mr-4 
                  ${currentQuestionIndex === 0 ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
              >
                <ArrowLeft className="h-6 w-6" />
              </button>
              <h2 className="text-2xl font-bold text-white">
                {surveyQuestions[currentQuestionIndex]?.question}
              </h2>
            </div>
            
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-8">
              {surveyQuestions[currentQuestionIndex]?.choices.map((film) => (
                <div 
                  key={film.movieid} 
                  onClick={() => handleMovieSelection(film.movieid)} // Film seçildiğinde ID'yi kaydet
                  className="group bg-white/90 backdrop-blur-sm rounded-xl overflow-hidden border border-gray-200 transition-all duration-300 hover:shadow-xl hover:shadow-blue-500/10 cursor-pointer hover:scale-105"
                >
                  <div className="relative h-48 m-3 rounded-lg bg-gradient-to-br from-gray-100 to-gray-50 flex items-center justify-center">
                  </div>
                  <div className="p-6">
                    <h2 className="text-xl font-bold text-gray-800">
                      {film.title}
                    </h2>
                  </div>
                  <div className="p-2 ml-4 mb-6">
                    <p className="text-s text-gray-800">
                      {film.genres}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  );
};

export default SurveyPage;
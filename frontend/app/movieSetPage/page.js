"use client"; // Client Component direktifi

import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';

const PageLayout = () => {
  const router = useRouter();
  const [allMovies, setAllMovies] = useState([]); // Tüm filmleri saklamak için state
  const [currentPage, setCurrentPage] = useState(1); // Mevcut sayfa numarası
  const [isLoading, setIsLoading] = useState(true); // Yükleme durumu
  const [visiblePages, setVisiblePages] = useState([]); // Görünür sayfa numaraları
  const itemsPerPage = 6; // Sayfa başına gösterilecek film sayısı

  // API'den tüm filmleri çekme
  const fetchMovies = async () => {
    setIsLoading(true); // Yükleme başladı
    try {
      const response = await fetch('http://127.0.0.1:5000/api/movies/all-movies');
      const data = await response.json();
      if (Array.isArray(data)) {
        setAllMovies(data); // Tüm filmleri state'e kaydet
      } else {
        console.error('Unexpected API response:', data);
        setAllMovies([]); // Hata durumunda boş dizi ayarla
      }
    } catch (error) {
      console.error('Error fetching movies:', error);
      setAllMovies([]); // Hata durumunda boş dizi ayarla
    } finally {
      setIsLoading(false); // Yükleme tamamlandı
    }
  };

  // Sayfa değiştiğinde filmleri yeniden çek
  useEffect(() => {
    fetchMovies();
  }, []);

  // Sayfa değiştirme fonksiyonu
  const handlePageChange = (newPage) => {
    setCurrentPage(newPage);
    updateVisiblePages(newPage);
  };

  // Görünür sayfa numaralarını güncelle
  const updateVisiblePages = (page) => {
    const totalPages = Math.ceil(allMovies.length / itemsPerPage);
    let startPage = Math.max(1, page - 4); // Mevcut sayfadan önceki 4 sayfa
    let endPage = Math.min(totalPages, page + 4); // Mevcut sayfadan sonraki 4 sayfa

    // İlk 9 sayfayı göster
    if (page <= 5) {
      startPage = 1;
      endPage = Math.min(9, totalPages);
    }
    // Son 9 sayfayı göster
    else if (page >= totalPages - 4) {
      startPage = Math.max(1, totalPages - 8);
      endPage = totalPages;
    }

    const pages = [];
    for (let i = startPage; i <= endPage; i++) {
      pages.push(i);
    }
    setVisiblePages(pages);
  };

  // Mevcut sayfadaki filmleri hesapla
  const startIndex = (currentPage - 1) * itemsPerPage;
  const endIndex = startIndex + itemsPerPage;
  const currentMovies = allMovies.slice(startIndex, endIndex); // Mevcut sayfadaki filmler
  const totalPages = Math.ceil(allMovies.length / itemsPerPage); // Toplam sayfa sayısı

  // Görünür sayfa numaralarını başlangıçta ayarla
  useEffect(() => {
    updateVisiblePages(currentPage);
  }, [allMovies]);

  return (
    <div className="min-h-screen bg-gradient-animated">
      {/* Navbar */}
      <header className="max-w-7xl mx-auto px-4 py-8">
        <nav className="flex justify-between items-center">
          <Link href="/" className="text-2xl font-bold text-white">MovieMagic</Link>
          <div className="flex items-center space-x-6">
            <div className="relative">
            </div>
            <Link
              href="/surveyPage"
              className="text-white hover:bg-opacity-30 transition-colors duration-200 text-lg font-medium bg-white bg-opacity-15 rounded-full px-6 py-1"
            >
              Survey
            </Link>
          </div>
        </nav>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-12">
        {isLoading ? (
          // Yükleme durumu
          <div className="flex justify-center items-center h-64">
            <p className="text-white text-lg">Loading...</p>
          </div>
        ) : (
          // Filmlerin gösterimi
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-8">
            {currentMovies.map((movie) => (
                <div
                  key={movie.movieid}
                  className="group bg-white/90 backdrop-blur-sm rounded-xl overflow-hidden border border-gray-200 transition-all duration-300 hover:shadow-xl hover:shadow-blue-500/10 cursor-pointer hover:scale-[1.02]"
                  onClick={() => {
                    router.push(`/moviePage/${movie.movieid}`); // Film ID'sine göre yönlendirme
                  }}
                >
                  <div className="relative h-48 overflow-hidden m-3 rounded-lg">
                    <img
                      src={`https://via.placeholder.com/400x300?text=${movie.title}`} // Placeholder resmi
                      alt={movie.title}
                      className="w-full h-full object-cover transform group-hover:scale-105 transition-transform duration-300"
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-black/30 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                  </div>
                  <div className="p-6">
                    <h2 className="text-xl font-bold text-gray-800 mb-2">
                      {movie.title}
                    </h2>
                    <p className="text-black opacity-60 text-sm line-clamp-2">
                      {movie.genres}
                    </p>
                  </div>
                </div>
              ))}
          </div>
        )}
      </main>

      {/* Pagination */}
      <div className="flex justify-center items-center space-x-3 py-8">
        <button
          onClick={() => handlePageChange(currentPage - 1)}
          disabled={currentPage === 1 || isLoading}
          className="px-4 py-2 rounded-md bg-white/80 text-black font-medium hover:bg-blue-50 transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Previous
        </button>
        <div className="flex space-x-2">
          {visiblePages.map((page) => (
            <button
              key={page}
              onClick={() => handlePageChange(page)}
              disabled={isLoading}
              className={`px-4 py-2 rounded-md transition-colors duration-200 ${
                page === currentPage
                  ? 'bg-black bg-opacity-50 text-white font-medium'
                  : 'bg-white/80 text-black hover:bg-blue-50 font-medium'
              }`}
            >
              {page}
            </button>
          ))}
          {totalPages > 9 && visiblePages[visiblePages.length - 1] < totalPages && (
            <button
              onClick={() => handlePageChange(visiblePages[visiblePages.length - 1] + 1)}
              disabled={isLoading}
              className="px-4 py-2 rounded-md bg-white/80 text-black font-medium hover:bg-blue-50 transition-colors duration-200"
            >
              ...
            </button>
          )}
        </div>
        <button
          onClick={() => handlePageChange(currentPage + 1)}
          disabled={currentPage === totalPages || isLoading}
          className="px-4 py-2 rounded-md bg-white/80 text-black font-medium hover:bg-blue-50 transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Next
        </button>
      </div>
    </div>
  );
};

export default PageLayout;
'use client';

import { motion } from 'framer-motion';
import { FilmIcon, SparklesIcon, ThumbsUpIcon } from 'lucide-react';
import Image from 'next/image';
import Link from 'next/link';

const features = [
  { icon: FilmIcon, title: 'Vast Movie Database', description: 'Access to thousands of movies across all genres' },
  { icon: ThumbsUpIcon, title: 'Personalized Recommendations', description: 'Get movie suggestions tailored to your preferences' },
  { icon: SparklesIcon, title: 'Discover Hidden Gems', description: "Uncover lesser-known movies you'll love" },
];

const movies = [
  { title: 'Inception', poster: '/poster.jpg' },
  { title: 'The Shawshank Redemption', poster: '/poster.jpg' },
  { title: 'The Dark Knight', poster: '/poster.jpg' },
  { title: 'Pulp Fiction', poster: '/poster.jpg' },
];

export default function LandingPage() {

  return (
    <div className="min-h-screen bg-gradient-animated">
      {/* Header */}
      <header className="container mx-auto px-4 py-8">
        <nav className="flex justify-between items-center">
          <Link href="/" className="text-2xl font-bold text-white">MovieMagic</Link>
        </nav>
      </header>

      {/* Main Content */}
      <main>
        {/* Hero Section */}
        <section className="min-h-screen flex items-center justify-center text-white">
          <div className="container mx-auto px-4 flex flex-col space-y-8">
            <motion.h1 
              className="text-8xl font-extrabold drop-shadow-2xl"
              initial={{ opacity: 0, y: -50 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
            >
              Discover Your Next Favorite Movie
            </motion.h1>
            <motion.p 
              className="text-xl mt-4 drop-shadow-2xl"
              initial={{ opacity: 0, y: 50 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
            >
              Personalized recommendations based on your taste
            </motion.p>
            <motion.button 
              className="bg-white text-gray-800 text-xl font-bold py-2 px-3  rounded-xl hover:bg-blue-100 transition-colors mt-8 max-w-xs w-full"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <Link href="/crossroadPage">
              Get Started
              </Link>
            </motion.button>
          </div>
        </section>

        {/* Features Section */}
        <section className="min-h-screen flex items-center bg-gray-300 bg-opacity-10 rounded-3xl m-5 shadow-2xl">
          <div className="py-5 w-full">
            <div className="container mx-auto px-4">
              <h2 className="text-7xl font-bold text-center mb-20 text-white drop-shadow-2xl">Why Choose MovieMagic</h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                {features.map((feature, index) => (
                  <motion.div 
                    key={index}
                    className="text-center  rounded-xl p-8 "
                    initial={{ opacity: 0, y: 50 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: index * 0.1 }}
                    
                  >
                    <feature.icon className="w-12 h-12 mx-auto mb-4 text-white" />
                    <h3 className="text-2xl font-semibold mb-2 text-white">{feature.title}</h3>
                    <p className="text-gray-200 text-lg font-medium">{feature.description}</p>
                  </motion.div>
                ))}
              </div>
            </div>
          </div>
        </section>

        {/* Popular Movies Section */}
        <section className="py-20 bg-gray-100 rounded-3xl m-5">
          <div className="container mx-auto px-4 ">
            <h2 className="text-3xl font-bold text-center mb-12">Popular Movies</h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-8 ">
              {movies.map((movie, index) => (
                <motion.div 
                  key={index}
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  whileHover={{ scale: 1.05 }}
                  className="cursor-pointer"
                >
                  <Image 
                    src={movie.poster} 
                    alt={movie.title} 
                    width={300} 
                    height={400} 
                    className="rounded-3xl shadow-lg"
                  />
                  <div className='w-[300]'>
                  <h3 className="text-center mt-5 font-semibold">{movie.title}</h3>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </section>

        {/* Call To Action Section */}
        <section className="py-20 bg-gray-300 bg-opacity-5 shadow-2xl text-white text-center m-5 rounded-3xl">
          <div className="container mx-auto px-4">
            <motion.h2 
              className="text-3xl font-bold mb-4 drop-shadow-2xl"
              initial={{ opacity: 0, y: -50 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
            >
              Ready to Find Your Next Favorite Movie?
            </motion.h2>
            <motion.p 
              className="text-xl mb-8 drop-shadow-2xl"
              initial={{ opacity: 0, y: 50 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
            >
              Join MovieMagic today and start discovering amazing films!
            </motion.p>
            
            
            <motion.button 
              className="bg-white text-black font-bold py-4 px-8 rounded-full hover:bg-blue-100 transition-colors "
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <Link href='/crossroadPage'>
              Let's Start
              </Link>
            </motion.button>
            
          </div>
        </section>

      </main>

      {/* Footer */}
      <footer className="container mx-auto px-5 pt-5 pb-10  text-center text-white">
        <p>&copy; 2024 MovieMagic. All rights reserved.</p>
      </footer>
    </div>
  );
}

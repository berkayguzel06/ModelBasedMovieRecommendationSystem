"use client";

import { motion } from "framer-motion";
import Link from "next/link";

export default function CrossroadPage() {
  return (
    <div className="h-screen bg-gradient-animated flex justify-center">
          <motion.div
            className="bg-gray-300 bg-opacity-5 p-8 rounded-xl shadow-lg text-center  content-center my-10 ml-10 mr-5 flex-1"
            initial={{ opacity: 0, x: -50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5 }}
          >
            <h3 className="text-7xl text-white font-bold mb-4 drop-shadow-2xl">Start with custom preferences</h3>
            <p className="text-lg mb-8 mt-4 text-white font-semibold drop-shadow-2xl">
              Take a quick survey to get personalized movie recommendations.
            </p>
            <motion.button
              className="bg-black bg-opacity-20 text-white font-bold py-4 px-6 rounded-full hover:bg-opacity-30 transition-colors shadow-xl"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <Link href="/surveyPage">Take Survey</Link>
            </motion.button>
          </motion.div>
          <motion.div
            className="bg-gray-300 bg-opacity-5 p-8 rounded-xl shadow-lg text-center  content-center my-10 mr-10 ml-5 flex-1"
            initial={{ opacity: 0, x: -50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5 }}
          >
            <h3 className="text-7xl text-white font-bold mb-4 drop-shadow-2xl">Quick dive into popular movies</h3>
            <p className="text-lg mb-8 mt-4 text-white font-semibold drop-shadow-2xl">
              Dive into our extensive movie database and find your next watch.
            </p>
            <motion.button
              className="bg-black bg-opacity-20 text-white font-bold py-4 px-6 rounded-full hover:bg-opacity-30 transition-colors shadow-xl"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <Link href="/movieSetPage">Quick Dive</Link>
            </motion.button>
          </motion.div>
        
      
    </div>
  );
}

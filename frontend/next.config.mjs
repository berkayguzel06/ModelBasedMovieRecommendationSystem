/** @type {import('next').NextConfig} */
const nextConfig = {
    async rewrites() {
        return [
          {
            source: '/movieSetPage',
            destination: '/moviePage/[movieId]',
          },
        ];
      },
      reactStrictMode: true,
};

export default nextConfig;

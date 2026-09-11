import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* Vercel handles build output itself; no `output: standalone` here
     (that is only for a self-hosted Docker build and breaks Vercel's finalize step). */
};

export default nextConfig;

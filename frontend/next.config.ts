import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* Vercel handles build output itself; no `output: standalone` here
     (that is only for a self-hosted Docker build and breaks Vercel's finalize step). */

  /* Baseline security headers. The concrete risk for a crypto-adjacent page is being framed by a phishing site that
     overlays a fake "connect wallet" on top of our canvas, so SAMEORIGIN framing is the one that matters; the other
     two are free. No CSP here on purpose: the page streams from a separate WS/API origin
     (NEXT_PUBLIC_WS_URL / NEXT_PUBLIC_API_URL), so a useful `connect-src` has to be generated from those env vars -
     worth doing, but not as a silent default that breaks a self-hosted deploy pointing somewhere else. */
  async headers() {
    return [
      {
        source: "/:path*",
        headers: [
          { key: "X-Frame-Options", value: "SAMEORIGIN" },
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
        ],
      },
    ];
  },
};

export default nextConfig;

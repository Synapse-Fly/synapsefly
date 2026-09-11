// SPEC section e.7: the only server component. Favicon + social cards use the pixel-fly logo in /public.
import type { Metadata } from "next";
import "./globals.css";

const SITE = "https://www.synapsefly.com";
const TITLE = "SynapseFly ($SYNAPSE) — a fruit-fly brain trading on vibes";
const DESC =
  "A real fruit-fly connectome (~166k neurons) wired to the market, simulated live. " +
  "Buys feed it, sells scare it. It paints the price on a canvas and tweets its own brain.";

export const metadata: Metadata = {
  metadataBase: new URL(SITE),
  title: TITLE,
  description: DESC,
  icons: {
    icon: [{ url: "/favicon-48.png", type: "image/png", sizes: "48x48" }],
    apple: [{ url: "/apple-icon.png", sizes: "512x512" }],
  },
  openGraph: {
    type: "website",
    url: SITE,
    title: TITLE,
    description: DESC,
    siteName: "SynapseFly",
    images: [{ url: "/og.png", width: 1200, height: 630, alt: "SynapseFly $SYNAPSE" }],
  },
  twitter: {
    card: "summary_large_image",
    site: "@SynapseFly",
    creator: "@SynapseFly",
    title: TITLE,
    description: DESC,
    images: ["/og.png"],
  },
  alternates: { canonical: "/" },
  robots: { index: true, follow: true },
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en">
      <body className="bg-win-teal">{children}</body>
    </html>
  );
}

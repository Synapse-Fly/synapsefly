// SPEC section e.7: the only server component. Favicon + social cards use the pixel-fly logo in /public.
import type { Metadata, Viewport } from "next";
import "./globals.css";

const SITE = "https://www.synapsefly.com";
// Honesty: production runs a synthetic MaleCNS-shaped stand-in, not the real Janelia dataset (see docs/NOTICE.md
// section 1). The social card is the main channel, so it must say so too - not only the modal's fine print. That
// includes this title: on an X summary_large_image unfurl it is the headline printed directly under the card, so
// "synthetic" has to survive here as well as in DESC and in the card art. 68 chars, inside X's ~70-char display.
const TITLE = "SynapseFly ($SYNAPSE) — a synthetic fruit-fly brain trading on vibes";
const DESC =
  "A synthetic MaleCNS-shaped fly brain (20k spiking neurons, modeled on Janelia's ~166k-neuron connectome) " +
  "wired to the market. Buys feed it, sells scare it. It paints the price on a canvas and tweets its own brain.";

export const metadata: Metadata = {
  metadataBase: new URL(SITE),
  title: TITLE,
  description: DESC,
  icons: {
    icon: [
      { url: "/favicon-32.png", type: "image/png", sizes: "32x32" },
      { url: "/favicon-48.png", type: "image/png", sizes: "48x48" },
    ],
    apple: [{ url: "/apple-icon.png", sizes: "512x512" }],
  },
  manifest: "/manifest.webmanifest",
  // No `images` here on purpose: app/opengraph-image.tsx generates the 1200x630 card and Next injects
  // og:image/og:image:width/height/alt from it. Twitter then inherits those images because `twitter` declares no
  // `images` of its own (postProcessMetadata in next/dist/lib/metadata/resolve-metadata.js), so twitter:image
  // tracks the same generated card while the card type stays summary_large_image.
  openGraph: {
    type: "website",
    url: SITE,
    title: TITLE,
    description: DESC,
    siteName: "SynapseFly",
  },
  twitter: {
    card: "summary_large_image",
    site: "@SynapseFly",
    creator: "@SynapseFly",
    title: TITLE,
    description: DESC,
  },
  alternates: { canonical: "/" },
  robots: { index: true, follow: true },
};

// Next 16 serves `theme-color` from the viewport export, not from `metadata` (see
// node_modules/next/dist/docs/01-app/03-api-reference/04-functions/generate-viewport.md). Win95 desktop teal, so
// mobile browser chrome matches the page instead of going default white - most of our traffic arrives from X.
export const viewport: Viewport = {
  themeColor: "#008080",
  colorScheme: "light",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en">
      <body className="bg-win-teal">{children}</body>
    </html>
  );
}

// SPEC section e.7: the only server component. No font import (offline-first); favicon = a 16 px pixel fly as an
// inline data: SVG; the body is the Win95 teal desktop.
import type { Metadata } from "next";
import "./globals.css";

// 16 x 16 pixel fly (crisp rects): brown body, red eyes, light-blue wings, dark legs.
const FLY_SVG =
  '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" shape-rendering="crispEdges">' +
  '<rect x="0" y="0" width="16" height="16" fill="#008080"/>' +
  '<rect x="2" y="3" width="5" height="3" fill="#c8dcff"/><rect x="9" y="3" width="5" height="3" fill="#c8dcff"/>' +
  '<rect x="3" y="10" width="4" height="2" fill="#c8dcff"/><rect x="9" y="10" width="4" height="2" fill="#c8dcff"/>' +
  '<rect x="6" y="1" width="4" height="3" fill="#3b2a1a"/>' +
  '<rect x="6" y="2" width="1" height="1" fill="#c02020"/><rect x="9" y="2" width="1" height="1" fill="#c02020"/>' +
  '<rect x="6" y="4" width="4" height="4" fill="#4a3620"/>' +
  '<rect x="6" y="8" width="4" height="6" fill="#3b2a1a"/>' +
  '<rect x="6" y="9" width="4" height="1" fill="#6b4a2a"/><rect x="6" y="11" width="4" height="1" fill="#6b4a2a"/>' +
  '<rect x="3" y="6" width="3" height="1" fill="#2a1a0a"/><rect x="10" y="6" width="3" height="1" fill="#2a1a0a"/>' +
  '<rect x="2" y="8" width="4" height="1" fill="#2a1a0a"/><rect x="10" y="8" width="4" height="1" fill="#2a1a0a"/>' +
  '<rect x="3" y="12" width="3" height="1" fill="#2a1a0a"/><rect x="10" y="12" width="3" height="1" fill="#2a1a0a"/>' +
  "</svg>";

const FAVICON = "data:image/svg+xml," + encodeURIComponent(FLY_SVG);

export const metadata: Metadata = {
  title: "FlyBrain - CRUISING",
  description: "SynapseFly ($FLY): a spiking fly brain painting the market",
  icons: { icon: FAVICON },
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en">
      <body className="bg-win-teal">{children}</body>
    </html>
  );
}

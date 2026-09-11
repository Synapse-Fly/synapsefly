// 404 — a small Win95 dialog so a wrong URL still looks on-brand.
import Link from "next/link";

export default function NotFound() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-win-teal p-4">
      <div className="bevel-out w-full max-w-[420px]" style={{ boxShadow: "2px 2px 0 #000" }}>
        <div className="titlebar flex items-center gap-2 text-[13px]">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src="/favicon-48.png" alt="" width={16} height={16} style={{ imageRendering: "pixelated" }} />
          <span className="flex-1">SynapseFly — page not found</span>
        </div>
        <div className="bg-win-gray p-4 text-[13px]">
          <div className="flex items-start gap-3">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src="/logo-256.png" alt="" width={64} height={64} style={{ imageRendering: "pixelated" }} className="bevel-in bg-win-teal p-1" />
            <div>
              <div className="text-[15px] font-bold">404 — the fly wandered off.</div>
              <p className="mt-1">This page doesn&apos;t exist. The brain is still firing on the home page.</p>
            </div>
          </div>
          <div className="mt-4 flex flex-wrap gap-2">
            <Link href="/" className="btn95 flex h-[24px] items-center px-3 font-bold">← Back to the canvas</Link>
            <a href="https://x.com/SynapseFly" target="_blank" rel="noopener noreferrer" className="btn95 flex h-[24px] items-center px-3">𝕏 @SynapseFly</a>
          </div>
        </div>
      </div>
    </div>
  );
}

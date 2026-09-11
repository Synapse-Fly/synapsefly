// The social card. This file IS the card: Next renders it at build time through the App Router's
// `opengraph-image` file convention (node_modules/next/dist/docs/01-app/03-api-reference/03-file-conventions/
// 01-metadata/opengraph-image.md), so it is reproducible - edit the copy here, rebuild, done. The old hand-made
// /og.png had no generator and claimed "a real fruit-fly brain", which the honesty rule forbids (docs/NOTICE.md);
// it is now deleted rather than kept as a copy of this output, because a hand-copied duplicate drifts and the
// stale one is what gets crawled. Palette, assets and regeneration notes live in docs/BRAND.md.
import { ImageResponse } from "next/og";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { PROJECT, TOKEN } from "@/lib/brand";

export const alt = `${PROJECT} ($${TOKEN}) - a synthetic fruit-fly brain, trading on vibes`;
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

/** Sampled straight out of the previous og.png so the card keeps its look. */
const BG = "#090e0c"; // near-black desktop
const GRID = "#14201c"; // faint 40px graph paper
const GREEN = "#00ff66"; // $SYNAPSE green
const WHITE = "#ebf0ee";
const GREY = "#96aaa0"; // footer line

const GRID_STEP = 40;

// The fly is drawn 1:1 from an asset that is ALREADY at the card's size. That is deliberate and load-bearing:
// satori has no support for `image-rendering` (its only occurrence in node_modules/next/dist/compiled/@vercel/og
// is a camelCase->kebab SVG attribute-name table, and the <image> it emits carries only `filter` in its style), so
// an `imageRendering: "pixelated"` style is silently dropped and resvg falls back to bilinear. Scaling logo-256.png
// up here therefore feathered every hard edge and smeared the dithered wings - a visible regression on hard-edged
// pixel art. Any scaling has to happen offline, with nearest-neighbour; see docs/BRAND.md for the one command.
const LOGO_W = 390;
const LOGO_H = 382;
/** Pixel-art fly, embedded as a data URL. Read once at module scope (build time) - see the doc above. */
const logoBytes = await readFile(join(process.cwd(), "public", "logo-390.png"));
// A PNG's IHDR carries width/height as big-endian u32s at byte 16 and 20. Check them, so the day someone
// re-exports the asset at another size the build fails loudly instead of quietly shipping a resampled fly.
const assetW = logoBytes.readUInt32BE(16);
const assetH = logoBytes.readUInt32BE(20);
if (assetW !== LOGO_W || assetH !== LOGO_H) {
  throw new Error(
    `public/logo-390.png is ${assetW}x${assetH} but the card draws it at ${LOGO_W}x${LOGO_H}. They must match, ` +
      `or satori resamples the pixel art. Regenerate the asset (docs/BRAND.md) or update LOGO_W/LOGO_H.`,
  );
}
const LOGO_SRC = `data:image/png;base64,${logoBytes.toString("base64")}`;

/**
 * Satori breaks a text node into wrappable words and pads the space that precedes a hyphenated word, which makes
 * "A synthetic  fruit-fly  brain," look gappy. These lines are hand-broken anyway, so glue each one together with
 * non-breaking spaces and let it lay out as a single run.
 */
const oneRun = (s: string) => s.replace(/ /g, "\u00a0");

export default function Image() {
  const columns = Array.from({ length: Math.ceil(size.width / GRID_STEP) }, (_, i) => i * GRID_STEP);
  const rows = Array.from({ length: Math.ceil(size.height / GRID_STEP) }, (_, i) => i * GRID_STEP);

  return new ImageResponse(
    (
      <div
        style={{
          display: "flex",
          width: "100%",
          height: "100%",
          background: BG,
          color: WHITE,
          position: "relative",
        }}
      >
        {/* graph-paper grid */}
        {columns.map((x) => (
          <div
            key={`c${x}`}
            style={{ position: "absolute", left: x, top: 0, width: 1, height: size.height, background: GRID }}
          />
        ))}
        {rows.map((y) => (
          <div
            key={`r${y}`}
            style={{ position: "absolute", left: 0, top: y, width: size.width, height: 1, background: GRID }}
          />
        ))}

        {/* left: the fly. 504 wide and 630 tall centre a 390x382 image on whole pixels - (504-390)/2 = 57,
            (630-382)/2 = 124 - so the blit is the identity transform and every art pixel survives. Keep both
            LOGO_W and LOGO_H even, and do NOT add an `imageRendering` style here: it does nothing (see above). */}
        <div style={{ display: "flex", width: 504, alignItems: "center", justifyContent: "center" }}>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={LOGO_SRC} alt="" width={LOGO_W} height={LOGO_H} />
        </div>

        {/* right: the words */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            justifyContent: "center",
            flexGrow: 1,
            flexBasis: 0,
            paddingRight: 56,
          }}
        >
          <div
            style={{
              fontSize: 116,
              fontWeight: 700,
              lineHeight: 1.05,
              color: GREEN,
              letterSpacing: 1,
              WebkitTextStroke: `3px ${GREEN}`,
            }}
          >
            {`$${TOKEN}`}
          </div>
          <div
            style={{
              fontSize: 58,
              fontWeight: 700,
              lineHeight: 1.2,
              marginTop: 14,
              WebkitTextStroke: `1.6px ${WHITE}`,
            }}
          >
            {PROJECT}
          </div>
          {/* The honest line. Keep "synthetic" - see docs/NOTICE.md section 1. */}
          <div style={{ display: "flex", flexDirection: "column", marginTop: 14, fontSize: 34, lineHeight: 1.24 }}>
            <div>{oneRun("A synthetic fruit-fly brain,")}</div>
            <div>{oneRun("trading on vibes.")}</div>
          </div>
          <div style={{ marginTop: 38, fontSize: 23, color: GREY }}>
            {oneRun("20k spiking neurons · modeled on Janelia's MaleCNS")}
          </div>
        </div>
      </div>
    ),
    { ...size },
  );
}

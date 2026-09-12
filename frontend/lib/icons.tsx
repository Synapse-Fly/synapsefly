// Windows-95 pixel icon set, drawn as inline SVG (no PNG, no remote images, no deps).
//
// Grid
//   Every icon is a hand-placed 16 x 16 pixel bitmap rendered through a `viewBox="0 0 16 16"` SVG, so one art unit is
//   one logical pixel. 16 is the only grid that divides evenly into the three sizes the app actually uses - 16 px
//   (taskbar buttons and window title bars), 32 px and 48 px (the desktop). No viewBox larger than 8 can scale
//   integrally to 16/24/32/48 at once, so 24 px lands on a 1.5x scale; `shapeRendering="crispEdges"` snaps every
//   rect edge to a whole device pixel there, which keeps the art hard-edged (a mix of 1 px and 2 px runs) instead of
//   blurring it. That is exactly what Windows did when it stretched a 16 x 16 icon.
//
// Palette
//   The classic 16-colour VGA palette, plus ONE extra: the SynapseFly brand green sampled from public/logo-256.png,
//   which is used only on the fly ("status"). Light source is top-left, every shape carries a 1 px near-black
//   outline, no anti-aliasing anywhere.
//
// Encoding
//   Each icon is 16 strings of 16 chars, one char per palette slot ('.' = transparent). At module load the rows are
//   run-length encoded into horizontal <rect> runs (ICON_RUNS), so a 48 px desktop icon is ~60 rects, computed once.
import type { ReactElement } from "react";

export type IconName =
  | "paint" | "oscilloscope" | "status" | "brain" | "notepad" | "contract" | "x" | "github" | "readme" | "bin";

export const ICON_NAMES: readonly IconName[] = [
  "paint", "oscilloscope", "status", "brain", "notepad", "contract", "x", "github", "readme", "bin",
] as const;

/** Art grid edge, in art units. Also the SVG viewBox size. */
export const ICON_GRID = 16;

const PALETTE: Readonly<Record<string, string>> = {
  k: "#000000", g: "#808080", c: "#c0c0c0", w: "#ffffff",
  n: "#000080", b: "#0000ff", t: "#008080", y: "#00ffff",
  m: "#800000", r: "#ff0000", o: "#808000", e: "#ffff00",
  d: "#008000", l: "#00ff00", p: "#800080", f: "#ff00ff",
  // brand green, sampled from public/logo-256.png - the fly only.
  F: "#22e06a",
};

const ART: Readonly<Record<IconName, readonly string[]>> = {
  // Artist's palette with four wet blobs and a thumb hole, loaded brush coming in from the top-right.
  paint: [
    "...........kk...",
    "..........kowk..",
    ".........koook..",
    "........kook....",
    ".......kcck.....",
    "......kcck......",
    ".....krrk.......",
    "..kkkkk.........",
    ".kwwwwwk........",
    "kwrrweewwwk.....",
    "kwrrweewbbwwk...",
    "kwwwwwwwbbwwk...",
    "kwddwwkkwwwgk...",
    ".kddwwkkwwggk...",
    "..kkkkkkkkkk....",
    "................",
  ],
  // Beige CRT, dark screen, bright green spike train over a dotted graticule.
  oscilloscope: [
    ".kkkkkkkkkkkkkk.",
    ".kwwwwwwwwwwwwk.",
    ".kwkkkkkkkkkkgk.",
    ".kwkklkkkkkkkgk.",
    ".kwkdldkdklkkgk.",
    ".kwkklkkkklkkgk.",
    ".kwkklklkklkkgk.",
    ".kwkklklkklkkgk.",
    ".kwkllllllllkgk.",
    ".kwkkkkkkkkkkgk.",
    ".kgggggggggggk..",
    ".kkkkkkkkkkkkk..",
    "......kggk......",
    "......kggk......",
    "...kccccccccck..",
    "................",
  ],
  // The mascot, reduced from public/logo-256.png: head-on fly, red compound eyes, cyan frons glint,
  // near-black body carrying bright neural tracery, wings out to both sides.
  status: [
    ".....k....k.....",
    ".....k....k.....",
    "...kkkkkkkkkk...",
    "...krrkyykrrk...",
    "...krrkkkkrrk...",
    "...kmrkkkkrmk...",
    "....kmmkkmmk....",
    "..kkkkFkkFkkkk..",
    ".kwwkkFFFFkkwwk.",
    "kwwwkkFkkFkkwwwk",
    "kwcckkkkkkkkccwk",
    ".kkkkFFFFFFkkkk.",
    "...kkkkkkkkkk...",
    "....kkFFFFkk....",
    "......kkkk......",
    "................",
  ],
  // A brain in profile, stem down: the one shape that reads as "brain" at 16 px. The previous art was a two-colour
  // point cloud, which at icon size was an unreadable splat and sat right next to the (also colourful) fly icon.
  // Magenta body, dark-purple sulci running as continuous curves (dashes read as noise) and bottom-right shading.
  brain: [
    "................",
    ".....kkkkkk.....",
    "...kkwwffffkk...",
    "..kfffpfffpffk..",
    ".kfffpfffffpffk.",
    ".kfffpfffffpffk.",
    "kfffffpfffpffpfk",
    "kfpffffpffpffpfk",
    "kfpfffpffffpfpfk",
    ".kfffpfffffppfk.",
    "..kffpffffppfk..",
    "...kffffpfppk...",
    "....kkfffpkk....",
    "......kfpk......",
    "......kkkk......",
    "................",
  ],
  // Notepad sheet: folded top-right corner, four ruled lines.
  notepad: [
    "................",
    "...kkkkkkk......",
    "...kwwwwckk.....",
    "...kwwwwcck.....",
    "...kwwwwccck....",
    "...kwwwwwwwwk...",
    "...kwwwwwwwwk...",
    "...kwggggggwk...",
    "...kwwwwwwwwk...",
    "...kwggggggwk...",
    "...kwwwwwwwwk...",
    "...kwggggggwk...",
    "...kwwwwwwwwk...",
    "...kwggggwwwk...",
    "...kkkkkkkkkk...",
    "................",
  ],
  // CA.txt: the contract-address document. Deliberately NOT the notepad sheet - at launch these two icons sit five
  // cells apart in the same column ("CA.txt" and "tweets.txt") and a visitor hunting for the address must be able to
  // tell them apart at a glance, so this sheet is square-cornered (no folded corner), sits one pixel left and up, and
  // carries a gold coin overlapping its bottom-right corner. The coin is the only saturated yellow on the desktop, so
  // at 16 px it is the thing you actually see; at 48 px it is a coin with an olive shadow on its lower-right.
  contract: [
    "..kkkkkkkkkk....",
    "..kwwwwwwwwk....",
    "..kwggggggwk....",
    "..kwwwwwwwwk....",
    "..kwggggggwk....",
    "..kwwwwwwwwk....",
    "..kwggggwwwk....",
    "..kwwwwwwwwk....",
    "..kwggggggwk....",
    "..kwwwwwwwkkk...",
    "..kwgggggkeeek..",
    "..kwwwwwkeeeeek.",
    "..kwgggwkeeeeok.",
    "..kkkkkkkeeeook.",
    ".........keook..",
    "..........kkk...",
  ],
  // X (formerly Twitter): white mark on the black tile.
  x: [
    "..kkkkkkkkkkkk..",
    ".kkkkkkkkkkkkkk.",
    "kkkkkkkkkkkkkkkk",
    "kkkwwkkkkkkwwkkk",
    "kkkkwwkkkkwwkkkk",
    "kkkkkwwkkwwkkkkk",
    "kkkkkkwwwwkkkkkk",
    "kkkkkkkwwkkkkkkk",
    "kkkkkkwwwwkkkkkk",
    "kkkkkwwkkwwkkkkk",
    "kkkkwwkkkkwwkkkk",
    "kkkwwkkkkkkwwkkk",
    "kkkwkkkkkkkkwkkk",
    "kkkkkkkkkkkkkkkk",
    ".kkkkkkkkkkkkkk.",
    "..kkkkkkkkkkkk..",
  ],
  // GitHub: the Octocat silhouette in white on the black tile.
  github: [
    "..kkkkkkkkkkkk..",
    ".kkkkkkkkkkkkkk.",
    "kkkwwkkkkkkwwkkk",
    "kkwwwwkkkkwwwwkk",
    "kkkwwwwwwwwwwkkk",
    "kkwwwwwwwwwwwwkk",
    "kkwkkwwwwwkkwwkk",
    "kkwwwwwwwwwwwwkk",
    "kkkwwwwwwwwwwkkk",
    "kwwwwwwwwwwwwkkk",
    "kwkwwwwwwwwwkkkk",
    "kkkkwwwwkwwwkkkk",
    "kkkkwwwkkwwwkkkk",
    "kkkkwwkkkkwwkkkk",
    ".kkkkkkkkkkkkkk.",
    "..kkkkkkkkkkkk..",
  ],
  // Text document with the blue information badge.
  readme: [
    "................",
    ".kkkkkkkkkk.....",
    ".kwwwwwwwwk.....",
    ".kwggggggwk.....",
    ".kwwwwwwwwk.....",
    ".kwggggggwk.....",
    ".kwwwwwwwwk.....",
    ".kwggggggkkkk...",
    ".kwwwwwwkbbbbbbk",
    ".kwwwwwkbbwwbbbk",
    ".kwwwwwkbbbbbbnk",
    ".kwwwwwkbbwwbbnk",
    ".kwwwwwkbbwwbbnk",
    ".kkkkkkkknnnnnnk",
    ".........kkkk...",
    "................",
  ],
  // Recycle Bin: silver cylinder under an overhanging lid, hollow green recycling triangle.
  bin: [
    "................",
    "..kkkkkkkkkkkk..",
    "..kccccccccgkk..",
    "..kkkkkkkkkkkk..",
    "...kcccccccgk...",
    "...kcccccccgk...",
    "...kcccllccgk...",
    "...kcclcclcgk...",
    "...kclcccclgk...",
    "....kllllllk....",
    "....kcccccgk....",
    "....kcccccgk....",
    "....kcccccgk....",
    "....kcccccgk....",
    ".....kkkkkk.....",
    "................",
  ],
};

/**
 * The classic Windows 95 shortcut badge: a white box with a black border and a black arrow pointing up-left, sitting
 * in the bottom-left corner. 7 x 7 art units so the arrow is still legible at 16 px.
 */
const SHORTCUT: readonly string[] = [
  "kkkkkkk",
  "kwwwwwk",
  "kwkkkwk",
  "kwkkwwk",
  "kwkwkwk",
  "kwwwwkk",
  "kkkkkkk",
];
const SHORTCUT_X = 0;
const SHORTCUT_Y = ICON_GRID - SHORTCUT.length;

/** Icons that are links out to the web, so they wear the shortcut badge unless the caller says otherwise. */
const SHORTCUT_BY_DEFAULT: readonly IconName[] = ["x", "github"];

interface Run { x: number; y: number; w: number; fill: string }

/** Horizontal run-length encode a char bitmap into <rect> runs. '.' is transparent. */
function encode(rows: readonly string[], ox = 0, oy = 0): Run[] {
  const out: Run[] = [];
  rows.forEach((row, ry) => {
    let x = 0;
    while (x < row.length) {
      const ch = row[x];
      let w = 1;
      while (x + w < row.length && row[x + w] === ch) w += 1;
      const fill = PALETTE[ch];
      if (fill) out.push({ x: ox + x, y: oy + ry, w, fill });
      x += w;
    }
  });
  return out;
}

const ICON_RUNS: Record<IconName, readonly Run[]> = ICON_NAMES.reduce((acc, name) => {
  acc[name] = encode(ART[name]);
  return acc;
}, {} as Record<IconName, readonly Run[]>);

const SHORTCUT_RUNS: readonly Run[] = encode(SHORTCUT, SHORTCUT_X, SHORTCUT_Y);

export interface AppIconProps {
  name: IconName;
  /** Rendered edge in CSS px. Crisp at 16 / 32 / 48; 24 stays hard-edged via shapeRendering. */
  size?: number;
  className?: string;
  /** Overlay the Win95 shortcut arrow. Defaults to true for the web-link icons ("x", "github"). */
  shortcut?: boolean;
  /** Accessible name. Omit (the default) to render the icon as decoration next to a real text label. */
  title?: string;
}

export function AppIcon({ name, size = 48, className, shortcut, title }: AppIconProps): ReactElement {
  const runs = ICON_RUNS[name] ?? ICON_RUNS.readme;
  const badge = shortcut ?? SHORTCUT_BY_DEFAULT.includes(name);
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width={size}
      height={size}
      viewBox={`0 0 ${ICON_GRID} ${ICON_GRID}`}
      shapeRendering="crispEdges"
      className={className}
      style={{ display: "block", imageRendering: "pixelated" }}
      role={title ? "img" : undefined}
      aria-label={title}
      aria-hidden={title ? undefined : true}
      focusable="false"
    >
      {title ? <title>{title}</title> : null}
      {runs.map((r, i) => (
        <rect key={`a${i}`} x={r.x} y={r.y} width={r.w} height={1} fill={r.fill} />
      ))}
      {badge
        ? SHORTCUT_RUNS.map((r, i) => (
            <rect key={`s${i}`} x={r.x} y={r.y} width={r.w} height={1} fill={r.fill} />
          ))
        : null}
    </svg>
  );
}

export default AppIcon;

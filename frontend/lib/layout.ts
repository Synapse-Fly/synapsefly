// Desktop tiling math for the Win95 desktop (app/page.tsx). Pure: no DOM, no React, no side effects, so it can be
// unit-tested / swept over viewport sizes. SPEC section e.5/e.7 froze the window rects as constants (PAINT_RECT et al);
// the user asked for a desktop that fills ANY viewport, so the rects are now fractions of the live viewport box.
//
// Geometry contract
//   * the desktop div is positioned `relative` and the rects below are absolute coordinates inside it;
//   * the usable area ("the box") is the viewport minus the 30 px taskbar (+4 px breathing room) and a DESK_MARGIN
//     border. The box is NEVER taller than that - a floor of MIN_BOX_H used to win on a short viewport and pushed
//     the bottom row of windows behind the fixed taskbar (1300 x 600 put 22 px of two windows under it and forced a
//     scrollbar on the untouched default layout);
//   * every rect is integer, >= its minimum and inside the box - EXCEPT when the windows that happen to be open
//     cannot fit their minimums in the box at all (five windows on a 600 px tall viewport want 584 px of column for
//     Paint + tweets.txt alone). Then the column overflows the box on purpose and page.tsx grows the desktop, so the
//     windows are reachable by scrolling instead of being clipped or squashed below their minimums;
//   * `deskW`/`deskH` are the desktop size the default layout needs (page.tsx grows the desktop past that when a
//     window is dragged out, which is what makes the desktop scroll instead of clipping - the "asagi kaymiyorlar"
//     bug).
//
// Shape: two columns up to THREE_COL_MIN_VW (left = Paint over tweets.txt, right = Oscilloscope / Fly Status /
// Fly Brain 3D), three columns above it (Paint over tweets.txt | Oscilloscope over Fly Status | Fly Brain 3D) so a
// 2560-3840 px monitor does not stretch one window absurdly wide.
//
// The Paint window is sized by its DOCUMENT, not by a fraction: the 800 x 500 bitmap is aspect-locked, so a Paint
// rect of a "nice" fraction leaves gray dead bands inside the window (up to 20 % of the workspace on a 21:9 screen -
// the same complaint as the teal dead field, one level down). `fitPaintColumn` therefore picks the Paint height its
// column width deserves (tweets.txt underneath absorbs the rest of the column), and when the column is still wider
// than the document can use it trims the width and hands it to the side column(s) - i.e. the trimmed space is given
// to ANOTHER WINDOW, never back to the desktop.
// The corollary, which is deliberate: when there is no other window to hand it to (Paint alone, or Paint alone in the
// last column) the rect is NOT trimmed. Paint fills its column and the leftover shows as gray workspace around the
// centred canvas, because the alternative is the teal dead field the user complained about in the first place -
// gray inside a window still reads as "the application", teal beside it reads as "the page is broken".

export type WindowId = "paint" | "tweets" | "raster" | "status" | "brain";

/** Canonical order. The tiler keeps it, so a subset never shuffles the windows that stayed open. */
export const WINDOW_IDS: readonly WindowId[] = ["paint", "tweets", "raster", "status", "brain"] as const;

export interface WinRect { x: number; y: number; w: number; h: number }

export interface DesktopLayout {
  /** Usable desktop area (absolute coords inside the desktop div). Drag clamping and "maximize" use this. */
  box: WinRect;
  /** 1, 2 or 3 tiled columns (1 only when a single window is open, or the box is too narrow to split). */
  cols: 1 | 2 | 3;
  /** Desktop size the default layout needs (taskbar excluded). */
  deskW: number;
  deskH: number;
  /** The windows that were tiled (the `open` option), in canonical order. */
  tiled: readonly WindowId[];
  /** A rect for every window: the tiled ones fill the box, the rest get their cascade rect. */
  win: Record<WindowId, WinRect>;
}

/** Taskbar is 30 px; 4 px more keeps a window edge off its bevel. */
export const TASKBAR_H = 34;
/** Teal border around the tiled windows. */
export const DESK_MARGIN = 8;
/** Gap between tiled windows. */
export const GAP = 6;
/** Stacked (mobile) desktop padding - the `p-2` on the flex column in page.tsx. */
export const STACK_PAD = 8;
/** At or above this viewport width the layout uses three columns instead of two. */
export const THREE_COL_MIN_VW = 2200;
/**
 * Left strip of the desktop the icon column owns, so a tiled window never buries the icons (DesktopIcons is 80 px
 * per cell + its own 8 px padding a side). page.tsx passes the measured value - one column on a tall screen, two
 * when the icons have to wrap - and this is only the single-column default.
 */
export const ICON_GUTTER_W = 96;
/** Classic cascade stagger for a window opened on its own, and how many steps before it starts over. */
export const CASCADE_STEP = 26;
export const CASCADE_MAX_SLOTS = 8;

// ------------------------------------------------------------------- Paint document geometry (shared with PaintWindow)

/** Gray margin around the document, the .bevel-in border (2 px a side) and the room the resize handles need. */
export const DOC_PAD = 6, DOC_FRAME = 4, DOC_HANDLE = 8;
/** Workspace pixels the document box can never use, per axis. */
export const DOC_INSET = 2 * DOC_PAD + DOC_FRAME + DOC_HANDLE;
/** hello.canvas is 800 x 500: the CSS box scales, the aspect never does. */
export const DOC_ASPECT = 800 / 500;
/**
 * Paint window size minus its document CSS box, per axis (measured, and stable at every viewport):
 *   W: 2 x 2 px window border + 58 px ToolPalette            (+ DOC_INSET)
 *   H: 2 x 2 px border + 22 title + 24 menu + 42 colours + 22 status bar   (+ DOC_INSET)
 */
export const PAINT_CHROME_W = 62, PAINT_CHROME_H = 114;
export const PAINT_FRAME_W = PAINT_CHROME_W + DOC_INSET;
export const PAINT_FRAME_H = PAINT_CHROME_H + DOC_INSET;
/** ToolPalette content height (measured 328 for its 8+3 rows of 24 px tools and the 44 px ink sample, +4 cushion):
 *  the Paint window may never be shorter than this plus its chrome or the palette clips. */
export const TOOL_PALETTE_H = 332;
/** What Paint's height growth always leaves for tweets.txt underneath it (~5 log lines + its chrome). */
export const TWEETS_KEEP_H = 140;

export const MIN_W: Record<WindowId, number> = { paint: 560, tweets: 420, raster: 340, status: 340, brain: 340 };
export const MIN_H: Record<WindowId, number> = {
  paint: TOOL_PALETTE_H + PAINT_CHROME_H,     // 446: never clip the tool palette
  tweets: 132, raster: 160, status: 190, brain: 210,
};
/**
 * How short a window may be squeezed when the viewport is not tall enough for the column's real minimums (see
 * `squeezeMins`). 96 px is the title bar plus ~74 px of body - every one of these scrolls its own content, so it is
 * cramped but whole. Paint is exempt: below MIN_H.paint its tool palette clips, which is a broken window, not a
 * small one.
 */
const SQUEEZE_FLOOR_H: Record<WindowId, number> = { paint: MIN_H.paint, tweets: 96, raster: 96, status: 96, brain: 96 };

/**
 * The box the two-column tiling WANTS. These are not floors on the box any more: a viewport smaller than this gets a
 * box the size it really has (so nothing hides behind the taskbar) and the columns overflow it - see computeDeskBox.
 */
export const MIN_BOX_W = MIN_W.paint + GAP + MIN_W.raster;
export const MIN_BOX_H = Math.max(
  MIN_H.paint + GAP + MIN_H.tweets,
  MIN_H.raster + MIN_H.status + MIN_H.brain + 2 * GAP,
);
/**
 * The only hard floors on the box: they exist so a degenerate viewport (0, NaN, a 320 px phone in desktop mode) still
 * produces a usable rect. Every real viewport is larger, so the box always equals the space the viewport has.
 */
export const HARD_MIN_BOX_W = MIN_W.paint;
export const HARD_MIN_BOX_H = MIN_H.raster;
const MIN_BOX_W_3 = MIN_W.paint + 2 * (GAP + MIN_W.raster);

/** Stacked (mobile) heights for the windows whose content scrolls internally; Paint is computed from the column. */
export const STACKED_H: Record<Exclude<WindowId, "paint">, number> = { raster: 300, status: 290, brain: 430, tweets: 190 };

function num(v: number, fallback: number): number {
  return Number.isFinite(v) && v > 0 ? Math.round(v) : fallback;
}
function clampNum(v: number, lo: number, hi: number): number {
  if (!Number.isFinite(v)) return lo;
  return v < lo ? lo : v > hi ? hi : v;
}

/**
 * Effective row minimums for a column that has `room` px to place them in. Normally the minimums stand; when the
 * viewport is too short for them (five windows on a 600 px tall screen want 584 px for Paint + tweets.txt alone) they
 * are scaled down proportionally instead of overflowing - a 2 px scrollbar with two windows parked behind the taskbar
 * is a worse answer than a tweets.txt that is 30 px shorter. Every window except Paint scrolls its own content, so a
 * squeezed one still works; Paint's floor is absolute because its tool palette would clip.
 */
function squeezeMins(ids: readonly WindowId[], room: number): number[] {
  const mins = ids.map((id) => MIN_H[id]);
  const sum = mins.reduce((a, b) => a + b, 0);
  if (!(room > 0) || sum <= room) return mins;
  const k = room / sum;
  const floors = ids.map((id, i) => Math.min(mins[i], SQUEEZE_FLOOR_H[id]));
  const out = mins.map((m, i) => Math.max(floors[i], Math.floor(m * k)));
  // A row that hit its floor can still push the column over: take the excess off whatever rows have slack left.
  let over = out.reduce((a, b) => a + b, 0) - room;
  for (let guard = 0; over > 0 && guard < 64; guard++) {
    let best = -1, slack = 0;
    for (let i = 0; i < out.length; i++) { const s = out[i] - floors[i]; if (s > slack) { slack = s; best = i; } }
    if (best < 0) break;                        // every row is on its floor: this column really cannot fit
    const take = Math.min(slack, over);
    out[best] -= take;
    over -= take;
  }
  return out;
}

/** Split `total` (minus the gaps) among weighted rows, honouring every minimum. Rows that cannot shrink overflow. */
function splitRows(total: number, ids: readonly WindowId[]): number[] {
  const n = ids.length;
  const room = total - GAP * (n - 1);
  const mins = squeezeMins(ids, room);
  const weights = ids.map((id) => ROW_WEIGHT[id]);
  const avail = Math.max(mins.reduce((a, b) => a + b, 0), room);
  const wsum = weights.reduce((a, b) => a + b, 0) || n;
  const out = weights.map((wt, i) => Math.max(mins[i], Math.round((avail * wt) / wsum)));
  let drift = avail - out.reduce((a, b) => a + b, 0);
  if (drift > 0) {
    let big = 0;
    for (let i = 1; i < n; i++) if (weights[i] > weights[big]) big = i;
    out[big] += drift;
  } else {
    for (let guard = 0; drift < 0 && guard < 64; guard++) {
      let best = -1, slack = 0;
      for (let i = 0; i < n; i++) { const s = out[i] - mins[i]; if (s > slack) { slack = s; best = i; } }
      if (best < 0) break;                      // every row already at its minimum: the column overflows the box
      const take = Math.min(slack, -drift);
      out[best] -= take;
      drift += take;
    }
  }
  return out.map((v) => Math.max(1, Math.round(v)));
}

/** The Paint height a column of `colW` px deserves: exactly the 8:5 document plus its chrome. */
export function paintHeightFor(colW: number): number {
  const docW = Math.max(1, Math.round(colW) - PAINT_FRAME_W);
  return Math.round(docW / DOC_ASPECT) + PAINT_FRAME_H;
}

/**
 * The left column (Paint over tweets.txt) for a column width of `colW` and a box height of `boxH`.
 * 1. Paint takes the height its document wants, capped so tweets.txt keeps TWEETS_KEEP_H;
 * 2. whatever width the document then cannot use is returned to the caller (the side columns absorb it);
 * 3. tweets.txt takes the rest of the column height - nothing is left empty.
 */
export function fitPaintColumn(colW: number, boxH: number): { paintW: number; paintH: number; tweetsH: number } {
  // tweets.txt gives its minimum back (down to its squeeze floor) when the column is too short for both: 446 + 6 +
  // 132 needs 584 px, and a 600 px tall viewport has 550 - without this the column overflows behind the taskbar.
  const tweetsMin = Math.max(SQUEEZE_FLOOR_H.tweets, Math.min(MIN_H.tweets, boxH - GAP - MIN_H.paint));
  const maxPaintH = Math.max(MIN_H.paint, Math.min(boxH - GAP - tweetsMin, boxH - GAP - TWEETS_KEEP_H));
  const paintH = clampNum(paintHeightFor(colW), MIN_H.paint, maxPaintH);
  const needW = Math.round((paintH - PAINT_FRAME_H) * DOC_ASPECT) + PAINT_FRAME_W;
  const paintW = clampNum(needW, Math.min(MIN_W.paint, colW), colW);
  const tweetsH = Math.max(tweetsMin, boxH - GAP - paintH);
  return { paintW: Math.round(paintW), paintH: Math.round(paintH), tweetsH: Math.round(tweetsH) };
}

/**
 * The usable desktop area for a viewport of `vw` x `vh` CSS px. `gutter` is the left strip the desktop icon column
 * owns: the tiling starts to the right of it, so an opened window never buries the icons the visitor clicks.
 *
 * The box is exactly the room the viewport has (never the room the five-window tiling would like): flooring it at
 * MIN_BOX_W/MIN_BOX_H made every viewport under 634 px tall scroll before the visitor touched anything and parked
 * the bottom row of windows behind the fixed taskbar.
 */
export function computeDeskBox(vw: number, vh: number, gutter = 0): WinRect {
  const g = Math.max(0, Math.round(Number.isFinite(gutter) ? gutter : 0));
  const w = Math.max(HARD_MIN_BOX_W, num(vw, 1280) - 2 * DESK_MARGIN - g);
  const h = Math.max(HARD_MIN_BOX_H, num(vh, 800) - TASKBAR_H - 2 * DESK_MARGIN);
  return { x: DESK_MARGIN + g, y: DESK_MARGIN, w, h };
}

// --------------------------------------------------------------------------------- tiling an arbitrary open subset

/** Row weight inside a column holding an arbitrary subset (the tuned weights of the historical right column). */
const ROW_WEIGHT: Record<WindowId, number> = { paint: 62, tweets: 30, raster: 29, status: 30, brain: 41 };
/** A side column is never thinner than this even when the fraction says so (readability, not a hard minimum). */
const SIDE_PREF_W = { two: 420, three: 430 } as const;
const SIDE_MAX_W = { two: 760, three: 700 } as const;

const colMinW = (col: readonly WindowId[]): number => col.reduce((m, id) => Math.max(m, MIN_W[id]), 1);

/** Split `ids` into `k` columns in canonical order, the remainder landing in the EARLIER columns. */
function spread(ids: readonly WindowId[], k: number): WindowId[][] {
  if (k <= 1) return ids.length ? [ids.slice()] : [];
  const base = Math.floor(ids.length / k), rem = ids.length % k;
  const out: WindowId[][] = [];
  let i = 0;
  for (let c = 0; c < k; c++) {
    const take = base + (c < rem ? 1 : 0);
    if (take > 0) out.push(ids.slice(i, i + take));
    i += take;
  }
  return out;
}

/**
 * Column assignment for a subset. Paint keeps its own column (with tweets.txt under it when there is room for both
 * and something else still needs a column of its own), which is what reproduces the historical five-window shape:
 * 2 cols -> [paint, tweets] | [raster, status, brain];  3 cols -> [paint, tweets] | [raster, status] | [brain].
 */
function partitionColumns(present: readonly WindowId[], k: number): WindowId[][] {
  if (k <= 1 || present.length <= 1) return present.length ? [present.slice()] : [];
  if (present[0] === "paint") {
    const head: WindowId[] = ["paint"];
    if (present[1] === "tweets" && present.length > k) head.push("tweets");
    return [head, ...spread(present.slice(head.length), k - 1)];
  }
  return spread(present, k);
}

/** Target widths per column; the LAST column is re-measured by the caller so it always ends on the box edge. */
function columnWidths(cols: readonly WindowId[][], box: WinRect): number[] {
  const n = cols.length;
  const mins = cols.map(colMinW);
  if (n <= 1) return [box.w];
  const avail = box.w - GAP * (n - 1);
  const out = new Array<number>(n).fill(0);
  if (cols[0][0] === "paint") {
    const frac = n >= 3 ? 0.22 : 0.34;
    const pref = n >= 3 ? SIDE_PREF_W.three : SIDE_PREF_W.two;
    const max = n >= 3 ? SIDE_MAX_W.three : SIDE_MAX_W.two;
    for (let i = 1; i < n; i++) {
      const lo = Math.max(mins[i], pref);
      out[i] = clampNum(Math.round(box.w * frac), lo, Math.max(lo, max));
    }
    const side = () => out.slice(1).reduce((a, b) => a + b, 0);
    // Paint takes what is left; when that is under its minimum the side columns give width back, down to THEIR
    // minimums (the historical `rightW = max(MIN_W.raster, box.w - GAP - MIN_W.paint)` guard, generalised).
    let need = mins[0] - (avail - side());
    for (let i = 1; i < n && need > 0; i++) {
      const give = Math.min(need, out[i] - mins[i]);
      out[i] -= give;
      need -= give;
    }
    out[0] = Math.max(1, avail - side());
  } else {
    const each = Math.floor(avail / n);
    for (let i = 0; i < n; i++) out[i] = Math.max(mins[i], each);
  }
  return out;
}

/** Place one column's windows; returns the width it actually used (Paint may hand width back to the side columns). */
function placeColumn(
  col: readonly WindowId[], x: number, colW: number, box: WinRect,
  out: Partial<Record<WindowId, WinRect>>, allowTrim: boolean,
): number {
  if (col.length === 0) return 0;
  if (col[0] === "paint") {
    // Paint is aspect-locked: it always fills its column's HEIGHT (no teal dead space), and hands back the width
    // its 8:5 document cannot use - but only when a column to its right can absorb it.
    if (col.length === 1) {
      const need = Math.round((box.h - PAINT_FRAME_H) * DOC_ASPECT) + PAINT_FRAME_W;
      const w = allowTrim ? clampNum(need, Math.min(MIN_W.paint, colW), colW) : colW;
      out.paint = { x, y: box.y, w, h: box.h };
      return w;
    }
    if (col.length === 2 && col[1] === "tweets") {
      const fit = fitPaintColumn(colW, box.h);
      const w = allowTrim ? fit.paintW : colW;
      out.paint = { x, y: box.y, w, h: fit.paintH };
      out.tweets = { x, y: box.y + fit.paintH + GAP, w, h: fit.tweetsH };
      return w;
    }
  }
  const hs = splitRows(box.h, col);
  let y = box.y;
  col.forEach((id, i) => {
    out[id] = { x, y, w: colW, h: hs[i] };
    y += hs[i] + GAP;
  });
  return colW;
}

export interface LayoutOptions {
  /** Which windows to tile. Default: all five. Everything else keeps its cascade rect in `win`. */
  open?: readonly WindowId[];
  /** Left strip reserved for the desktop icon column (see ICON_GUTTER_W). */
  gutter?: number;
}

/**
 * Tile the OPEN windows over the viewport, filling the box with no dead space. Pure; integer rects; never NaN, zero
 * or negative. With all five open this reproduces the historical two/three-column shape exactly.
 */
export function computeLayout(vw: number, vh: number, opts: LayoutOptions = {}): DesktopLayout {
  const box = computeDeskBox(vw, vh, opts.gutter ?? 0);
  const wanted = opts.open ?? WINDOW_IDS;
  const present = WINDOW_IDS.filter((id) => wanted.includes(id));

  // Column count: never more columns than windows, three only on a very wide box, and never so many that a column
  // would fall under its minimum width.
  const maxCols = num(vw, 1280) >= THREE_COL_MIN_VW && box.w >= MIN_BOX_W_3 ? 3 : 2;
  let cols: WindowId[][] = [];
  for (let k = Math.min(present.length, maxCols); k >= 1; k--) {
    cols = partitionColumns(present, k);
    const need = cols.reduce((a, c) => a + colMinW(c), 0) + GAP * (cols.length - 1);
    if (need <= box.w || k === 1) break;
  }

  const tiledRects: Partial<Record<WindowId, WinRect>> = {};
  const widths = columnWidths(cols, box);
  let x = box.x;
  cols.forEach((col, i) => {
    const last = i === cols.length - 1;
    // A non-last column may not eat the room the columns after it still need.
    const reserved = cols.slice(i + 1).reduce((a, c) => a + colMinW(c) + GAP, 0);
    const colW = last
      ? Math.max(colMinW(col), box.x + box.w - x)
      : Math.max(colMinW(col), Math.min(widths[i], box.x + box.w - x - reserved));
    x += placeColumn(col, x, colW, box, tiledRects, !last) + GAP;
  });

  // Windows that are not tiled still need a rect (they are cascading, or closed): give them their cascade rect.
  const win = {} as Record<WindowId, WinRect>;
  WINDOW_IDS.forEach((id, i) => { win[id] = tiledRects[id] ?? cascadeRect(box, i, id); });

  let deskW = box.x + box.w, deskH = box.y + box.h;
  for (const id of present) {
    const r = win[id];
    deskW = Math.max(deskW, r.x + r.w);
    deskH = Math.max(deskH, r.y + r.h);
  }
  const nCols = cols.length >= 3 ? 3 : cols.length === 2 ? 2 : 1;
  return { box, cols: nCols, deskW: deskW + DESK_MARGIN, deskH: deskH + DESK_MARGIN, tiled: present, win };
}

// ------------------------------------------------------------------------------------------------------- cascade

/** The size a window opened on its own gets: a readable fraction of the box, never under its minimum. */
export function defaultSize(box: WinRect, id: WindowId): { w: number; h: number } {
  if (id === "paint") {
    const w0 = clampNum(Math.round(box.w * 0.66), Math.min(MIN_W.paint, box.w), Math.min(box.w, 1060));
    const h = clampNum(paintHeightFor(w0), Math.min(MIN_H.paint, box.h), box.h);
    const need = Math.round((h - PAINT_FRAME_H) * DOC_ASPECT) + PAINT_FRAME_W;
    return { w: clampNum(need, Math.min(MIN_W.paint, box.w), Math.min(box.w, w0)), h };
  }
  const frac: Record<Exclude<WindowId, "paint">, readonly [number, number, number]> = {
    tweets: [0.46, 0.36, 680], raster: [0.48, 0.46, 760], status: [0.46, 0.48, 740], brain: [0.46, 0.62, 760],
  };
  const [wf, hf, wMax] = frac[id as Exclude<WindowId, "paint">];
  return {
    w: clampNum(Math.round(box.w * wf), Math.min(MIN_W[id], box.w), Math.min(box.w, wMax)),
    h: clampNum(Math.round(box.h * hf), Math.min(MIN_H[id], box.h), box.h),
  };
}

/**
 * The classic Win95 stagger: window number `slot` opens CASCADE_STEP px down and right of the previous one, and
 * starts over at the top-left once the box has no room left for another step (so nothing ever lands off-screen, and
 * two windows opened back to back never share a rect).
 */
export function cascadeRect(box: WinRect, slot: number, id: WindowId): WinRect {
  const { w, h } = defaultSize(box, id);
  const roomX = Math.max(0, box.w - w), roomY = Math.max(0, box.h - h);
  const steps = Math.max(1, Math.min(CASCADE_MAX_SLOTS, Math.floor(Math.min(roomX, roomY) / CASCADE_STEP) + 1));
  const n = Number.isFinite(slot) ? Math.round(slot) : 0;
  const s = ((n % steps) + steps) % steps;
  return { x: box.x + Math.min(roomX, s * CASCADE_STEP), y: box.y + Math.min(roomY, s * CASCADE_STEP), w, h };
}

/**
 * Stacked (mobile) rects: the width comes from CSS, only the height matters - except for Paint, whose height is the
 * one its document needs at the column width (a constant here left up to half the window empty on a phone), never
 * shorter than the tool palette. The other four scroll their own content, so their historical heights stand.
 */
export function computeStackedLayout(vw: number): Record<WindowId, WinRect> {
  const colW = Math.max(MIN_W.raster, num(vw, 390) - 2 * STACK_PAD);
  return {
    paint: { x: 0, y: 0, w: colW, h: Math.max(MIN_H.paint, paintHeightFor(colW)) },
    tweets: { x: 0, y: 0, w: colW, h: STACKED_H.tweets },
    raster: { x: 0, y: 0, w: colW, h: STACKED_H.raster },
    status: { x: 0, y: 0, w: colW, h: STACKED_H.status },
    brain: { x: 0, y: 0, w: colW, h: STACKED_H.brain },
  };
}

// ------------------------------------------------------------------------------- Paint document fit (PaintWindow)

/** The 800x500 bitmap is the protocol; only its CSS box scales. Within 2 % of an integer multiple we snap to it. */
export const DOC_SNAP_TOL = 0.02;
export const DOC_MIN_SCALE = 0.1;

/**
 * Largest aspect-correct scale for the 800x500 document inside a `boxW` x `boxH` content area.
 * >= 1: the exact fit, snapped down to an integer multiple when it is within DOC_SNAP_TOL of one (crisp 1x / 2x / 3x
 * pixel art when that costs under 2 % of the box - a wider tolerance threw away up to 6 % of the window the tiling
 * had just measured out for the document, which is exactly the dead space the user complained about).
 * < 1 (window smaller than the document): fractional, so the whole document still shows.
 */
export function fitDocScale(boxW: number, boxH: number, docW = 800, docH = 500): number {
  if (!(docW > 0) || !(docH > 0)) return 1;
  const bw = Number.isFinite(boxW) ? boxW : 0, bh = Number.isFinite(boxH) ? boxH : 0;
  const raw = Math.min(bw / docW, bh / docH);
  if (!Number.isFinite(raw) || raw <= 0) return DOC_MIN_SCALE;
  if (raw < 1) return Math.max(DOC_MIN_SCALE, raw);
  const k = Math.floor(raw + 1e-9);
  return raw - k <= DOC_SNAP_TOL * k ? k : raw;
}

/** `fitDocScale` as an integer CSS box (what PaintWindow puts on the .bevel-in document frame). */
export function fitDocBox(boxW: number, boxH: number, docW = 800, docH = 500): { w: number; h: number; scale: number } {
  const scale = fitDocScale(boxW, boxH, docW, docH);
  return { scale, w: Math.max(1, Math.round(docW * scale)), h: Math.max(1, Math.round(docH * scale)) };
}

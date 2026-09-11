# Brand assets

Card palette, sampled from the social card (the desktop itself uses the Win95 16-colour palette plus `#008080` teal):
`#090e0c` background · `#14201c` 40px grid · `#00ff66` $SYNAPSE green · `#ebf0ee` white text · `#96aaa0` footer grey.

Names, links and the token symbol are **not** duplicated here — `frontend/lib/brand.ts` is the single source of truth.

## Art in `frontend/public/`

| file | size | used by |
| --- | --- | --- |
| `logo.png` | 512x503 | master pixel-art fly |
| `logo-256.png` | 256x251 | the fly in the browser (IntroModal, 404 page) — CSS `image-rendering: pixelated` keeps it crisp there |
| `logo-390.png` | 390x382 | **derived** — the fly on the social card, see below |
| `favicon-32.png` / `favicon-48.png` | 32 / 48 | browser favicons; `favicon-48.png` is `LOGO` in `frontend/lib/brand.ts` (title-bar icon) |
| `apple-icon.png` | 512 | apple-touch-icon + webmanifest icon |

`logo-390.png` is a nearest-neighbour upscale of `logo-256.png`, regenerated with:

```sh
cd frontend && py -3 -c "from PIL import Image; Image.open('public/logo-256.png').convert('RGBA').resize((390,382), Image.NEAREST).save('public/logo-390.png', optimize=True)"
```

It exists because **satori ignores `image-rendering`**. The only occurrence of the property anywhere in
`node_modules/next/dist/compiled/@vercel/og` is a camelCase→kebab SVG attribute-name table, and the `<image>`
satori emits carries nothing but `filter` in its style, so an `imageRendering: "pixelated"` style on an `<img>` is
dropped without warning and resvg resamples bilinearly. On art with 1px outlines and dithered wings that reads as a
blurred logo. So the card never scales the fly: it embeds an asset that is *already* 390x382 and draws it at exactly
that size, on whole-pixel offsets, which makes the blit the identity transform. `opengraph-image.tsx` reads the
PNG's IHDR and throws at build time if the asset size and the drawn size ever diverge — if that error fires, either
regenerate the asset at the drawn size or change both together. Never reach for `image-rendering` to fix it.

## The social card

The 1200x630 card is **generated, not hand-drawn**: `frontend/app/opengraph-image.tsx` (next/og `ImageResponse` via
the App Router file convention) renders it at build time, Next serves it at `/opengraph-image?<hash>` and injects
`og:image` + `twitter:image` from it. There is no copy of it in `public/` — a hand-copied duplicate silently goes
stale, and the stale one is the artifact that gets crawled. `/og.png` was that duplicate and is gone; it is
deliberately a 404.

Edit the copy in `opengraph-image.tsx` to change the card, then `npm run build`. Nothing else to refresh.

**Keep it honest.** The brain is a *synthetic* MaleCNS-shaped stand-in modelled on Janelia's ~166k-neuron
connectome — never "a real fruit-fly brain" (`docs/NOTICE.md` section 1, `docs/PUPPETEERING.md`). That word has to
survive in all four places an X unfurl shows: the card art, `og:description`, `og:image:alt`, and `TITLE` in
`frontend/app/layout.tsx` — the title is the headline printed directly beneath the card, so it is the easiest one to
leave behind and the most prominent one to get wrong.

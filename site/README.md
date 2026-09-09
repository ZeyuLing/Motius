# Motius project homepage

Static GitHub Pages source: HTML, CSS and a small progressive-enhancement script.
No package manager, third-party fonts, UI framework, analytics or WebGL runtime.
Documentation links point to the repository's canonical Markdown guides.

## Design

Audience: motion researchers deciding whether to use the framework. Primary
action: open the installation guide / model zoo. Mode: marketing, with an
editorial emphasis. Visual thesis: a restrained blue/graphite identity and
large, attributable motion videos; native controls and complete text descriptions
remain available without JavaScript. The homepage does not promote unverified
pair-contact renders or imply every method supports training.

## Build and preview

```bash
# Only when the published source media changes:
python tools/prepare_project_site_media.py
# A new empty directory is required for each build:
python tools/build_project_site.py --output outputs/pages
python -m http.server 8080 --directory outputs/pages
```

The preparation command needs Pillow and imageio-ffmpeg; the build uses only
Python's standard library. Media provenance is in `site/media/provenance.json`.
The build copies an explicit allowlist, never the working tree, checkpoints,
private sources, or downloaded character models.

`.github/workflows/pages.yml` publishes `main` to GitHub Pages. Set the repository's
Pages source to **GitHub Actions**. Expected URL: https://zeyuling.github.io/Motius/.
Repository-relative assets also work under the project subpath.

## Verification

Check 320, 390, 768 and 1440 CSS-pixel widths, keyboard skip navigation, focus,
video playback, clipboard success/failure, reduced motion, failed video loading
and enlarged text. Native video controls provide pause and fullscreen. Only the
lead video may autoplay muted, once when visible; reduced-motion and data-saving
preferences suppress autoplay. Offscreen pages pause their videos.

Field Core Web Vitals and real screen-reader behavior require separate checks;
do not infer WCAG conformance from screenshots or automated tests alone.

Verified on 2026-09-09 in headless Chrome: all four viewport widths above,
keyboard skip navigation, actual clipboard copying, reduced-motion suppression,
enlarged text reflow and failed-media recovery. The deployed site returned HTTP
200, with all three MP4s decoded and played (5 s AutoRig, 3 s Mixamo, 6 s
representation conversion), no JavaScript page errors, and no mobile overflow.
The main muted-text and primary-button contrast ratios are 6.01:1 and 5.95:1.
Clipboard-denial messaging is implemented but was not included in the automated
deployment check. No screen-reader or field-performance conformance claim is made.

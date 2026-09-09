# Motius project homepage

Static GitHub Pages source: HTML, CSS and a small progressive-enhancement script.
No package manager, third-party fonts, UI framework, analytics or WebGL runtime.
Documentation links point to the repository's canonical Markdown guides.

## Design

Audience: motion researchers deciding whether to use the framework. Primary
action: open the installation guide / model zoo. Mode: marketing, with an
editorial emphasis. Visual thesis: a restrained blue/graphite identity and
attributable motion videos; native controls and complete text descriptions
remain available without JavaScript. The homepage does not promote unverified
pair-contact renders or imply every method supports training.

The reading order follows the framework's purpose, not the availability of
character-rendering assets:

1. Models and tasks: generation examples and a selected method directory.
2. Research workflow: a real inference API example and the public HYMotion
   training recipe, with data and hardware prerequisites.
3. Evaluation: HumanML3D Official, MotionStreamer, and Motius-trained universal
   TMR, followed by measured benchmark entry points and protocol boundaries.
4. Motion toolkit: representation conversion, Mixamo retargeting, and AutoRig.
5. Installation and documentation.

All three original toolkit videos are retained, with no motion or rendering
changes. HYMotion's 004545 jumping/clapping example and Bailando's AIST++
waacking example are existing Motius outputs, not newly run inference. The
HYMotion MP4 is a browser-compatible transcode preserving its source GIF's
2.85-second presentation timing; it does not repair the legacy GIF's frame-rate
quantization. Bailando's existing silent MP4 is copied unchanged. Original
model cards and audio-synchronized benchmarks remain linked. This page does
not advertise the unverified paired-contact FBX experiment.

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

Verified after rebalancing on 2026-09-09 in headless Chrome: all four viewport
widths above, keyboard skip navigation, actual clipboard copying and
clipboard-denial recovery, touch navigation, reduced-motion suppression,
enlarged text reflow and failed-media recovery. All five MP4s decoded and
played at each viewport (2.85 s HYMotion, 6 s Bailando, 6 s representation
conversion, 3 s Mixamo, 5 s AutoRig), with no JavaScript page errors or
horizontal page overflow. An enlarged-text overflow in the method directory
was fixed with shrinkable grid tracks and wrap-safe model links.
The main muted-text and primary-button contrast ratios are 6.01:1 and 5.95:1.
No screen-reader or field-performance conformance claim is made. The displayed
model/training snippets were checked against source contracts; this frontend
verification does not rerun GPU inference or distributed training.

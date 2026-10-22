# README release artwork

Regenerate with `python tools/render_readme_news.py` (requires fonttools).
Outfit glyph outlines use the existing font and SIL Open Font License in
`site/vendor/Outfit-OFL.txt`. These SVGs need no external fonts, scripts,
embedded images or browser-specific HTML. The desktop and mobile variants
contain the same release information; README links and alt text remain semantic.

FlowHMR also has generated left/right crops. Keep their README image links
adjacent without intervening whitespace and at 50% width each: they form one
continuous banner, while the left links to Motius documentation and the right
links to the original project. SVG links inside an image are not interactive
in GitHub README, so the two real HTML links are intentional.

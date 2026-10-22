"""Render GitHub-compatible release artwork with outlined Outfit typography.

Requires fonttools (only to regenerate the checked-in SVGs). No fonts, scripts,
foreignObject, embedded bitmaps or external resources are needed to display them.
"""
from html import escape
from pathlib import Path

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "assets/news"
FONT = ROOT / "site/vendor/Outfit.ttf"
FONTS = {
    weight: instantiateVariableFont(TTFont(FONT), {"wght": weight})
    for weight in (400, 550)
}


def text(value, x, baseline, size, color, weight=400, tracking=0):
    font = FONTS[weight]
    glyphs = font.getGlyphSet()
    cmap = font.getBestCmap()
    scale = size / font["head"].unitsPerEm
    pen = SVGPathPen(glyphs, ntos=lambda value: f'{value:.3f}'.rstrip('0').rstrip('.') if value else '0')
    for char in value:
        name = cmap[ord(char)]
        glyphs[name].draw(TransformPen(pen, (scale, 0, 0, -scale, x, baseline)))
        x += glyphs[name].width * scale + tracking
    return f'<path fill="{color}" d="{pen.getCommands()}"/>'


def arrow(x, y, color, size=18):
    return f'<path d="M{x-size} {y+size}L{x} {y}M{x-size} {y}H{x}V{y+size}" fill="none" stroke="{color}" stroke-width="2.3"/>'


def trajectories(width, height, mobile):
    x = 390 if mobile else 820
    pieces = ['<g clip-path="url(#bounds)" fill="none">']
    for i in range(14):
        offset = i * 16
        pieces.append(
            f'<path d="M{x+offset} {height+60} '
            f'C{x-140+offset} {height-70} {x+220+offset} 96 '
            f'{x+80+offset} -80" stroke="#8db9ff" stroke-opacity=".13"/>'
        )
    pieces.extend([
        f'<path d="M{x+82} {height+40}C{x-58} {height-66} {x+302} 86 {x+162} -80" stroke="url(#trail)" stroke-width="2"/>',
        f'<circle cx="{x+150}" cy="{height*.48}" r="5" fill="#a8d6ff" stroke="#173b82" stroke-width="8"/>',
        '</g>',
    ])
    return ''.join(pieces)


def render(key, date, title, description, mobile=False, featured=False):
    width = 600 if mobile else 1200
    height = (292 if mobile else 268) if featured else (156 if mobile else 122)
    ink, muted = ("#ffffff", "#c6d5ef") if featured else ("#172338", "#596b82")
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{escape(title)} — {escape(date)}</title>',
        f'<desc id="desc">{escape(description)} Open the Motius documentation.</desc>',
        '<defs><linearGradient id="paper" x2="1" y2="1"><stop stop-color="#10294e"/><stop offset="1" stop-color="#143e87"/></linearGradient>',
        '<linearGradient id="trail" x2="0" y2="1"><stop stop-color="#79b9ff" stop-opacity="0"/><stop offset=".5" stop-color="#9bd8ff"/><stop offset="1" stop-color="#79b9ff" stop-opacity="0"/></linearGradient>',
        f'<clipPath id="bounds"><rect width="{width}" height="{height}" rx="8"/></clipPath></defs>',
    ]
    if featured:
        pieces.append(f'<rect width="{width}" height="{height}" rx="8" fill="url(#paper)"/>')
        pieces.append(trajectories(width, height, mobile))
        x = 30 if mobile else 44
        pieces.extend([
            text(date, x, 48 if mobile else 48, 21 if mobile else 20, muted, tracking=.8),
            text(title, x-2, 139 if mobile else 145, 70 if mobile else 82, ink, 550, -2),
            text(description, x, 189 if mobile else 195, 25 if mobile else 29, muted),
            text('Explore the integration', x, 254 if mobile else 239, 20, ink),
            text('Original project', 315 if mobile else 865, 225 if mobile else 212, 18, muted),
            text('github.com/flowhmr/flowhmr', 315 if mobile else 865, 254 if mobile else 239, 17 if mobile else 20, ink),
            arrow(width-35, 35, ink),
        ])
    else:
        pieces.append(f'<rect width="{width}" height="{height}" rx="8" fill="#f5f7fb"/>')
        if mobile:
            pieces.extend([
                text(date, 30, 35, 19, muted, tracking=.5),
                text(title, 29, 86, 36, ink, 550, -.7),
                text(description, 30, 122, 21, muted),
                arrow(width-35, 65, "#315fba"),
            ])
        else:
            pieces.extend([
                text(date, 44, 54, 20, muted, tracking=.5),
                text(title, 295, 55, 36, ink, 550, -.7),
                text(description, 296, 90, 21, muted),
                arrow(width-36, 48, "#315fba"),
            ])
    pieces.append('</svg>')
    suffix = '-mobile' if mobile else ''
    artwork = '\n'.join(pieces)+'\n'
    (DEST / f'{key}{suffix}.svg').write_text(artwork)
    if featured:
        # GitHub displays SVGs as images, so embedded SVG anchors cannot be
        # clicked. Two adjacent image links preserve the continuous banner and
        # give the integration and original project independent real targets.
        for side, origin in [('left', 0), ('right', width // 2)]:
            cropped = artwork.replace(
                f'viewBox="0 0 {width} {height}"',
                f'viewBox="{origin} 0 {width // 2} {height}"',
                1,
            )
            if side == 'right':
                cropped = cropped.replace(
                    f'<title id="title">{escape(title)} — {escape(date)}</title>',
                    '<title id="title">Original FlowHMR project: github.com/flowhmr/flowhmr</title>',
                )
            (DEST / f'{key}{suffix}-{side}.svg').write_text(cropped)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    releases = (
        ('flowhmr', 'OCT 21, 2026', 'FlowHMR', 'Integrated into Motius.', True),
        ('autorig', 'AUG 23, 2026', 'AutoRig support', 'With Make-It-Animatable integration.', False),
        ('motius', 'AUG 11, 2026', 'Motius is open source', 'The first public release of the framework.', False),
    )
    for key, date, title, description, featured in releases:
        for mobile in (False, True):
            render(key, date, title, description, mobile=mobile, featured=featured)


if __name__ == '__main__':
    main()

import importlib.util
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit

import pytest

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location('project_site_builder', ROOT / 'tools/build_project_site.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_site_build_is_allowlisted_and_links_resolve(tmp_path):
    module.build(tmp_path)
    expected = {'index.html','style.css','app.js','.nojekyll','media/logo.png',
                'media/autorig.mp4','media/characters.mp4','media/representations.mp4',
                'media/autorig.jpg','media/characters.jpg','media/representations.jpg',
                'media/generation.mp4','media/generation.jpg','media/dance.mp4','media/dance.jpg'}
    assert {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob('*') if p.is_file()} == expected
    class Links(HTMLParser):
        def handle_starttag(self, tag, attrs):
            for key, value in attrs:
                if key in {'src','href','poster'} and value and not urlsplit(value).scheme and not value.startswith('#'):
                    assert (tmp_path / value).exists(), value
                prefix = 'https://github.com/ZeyuLing/Motius/blob/main/'
                if key == 'href' and value and value.startswith(prefix):
                    assert (ROOT / value[len(prefix):].split('#')[0]).is_file(), value
    Links().feed((tmp_path / 'index.html').read_text(encoding='utf-8'))


def test_site_build_never_deletes_existing_files(tmp_path):
    existing = tmp_path / 'user-file.txt'
    existing.write_text('preserve')
    with pytest.raises(ValueError, match='must be empty'):
        module.build(tmp_path)
    assert existing.read_text() == 'preserve'


def test_research_sections_precede_complete_character_toolkit():
    html = (ROOT / 'site/index.html').read_text(encoding='utf-8')
    sections = ['models', 'workflow', 'evaluation', 'motion-toolkit']
    positions = [html.index(f'id="{name}"') for name in sections]
    assert positions == sorted(positions)
    toolkit = html[positions[-1]:]
    for name in ('autorig', 'characters', 'representations'):
        assert f'src="media/{name}.mp4"' in toolkit
    assert html.count('data-autoplay') == 1
    assert 'data-autoplay' not in toolkit
    assert 'Universal TMR' in html and 'MotionStreamer' in html
    assert 'configs/hymotion_t2m/train_hymotion_t2m.py' in html


def test_homepage_fragment_targets_and_inference_example():
    import ast

    class Document(HTMLParser):
        def __init__(self):
            super().__init__()
            self.ids, self.fragments, self.example = [], [], []
            self.in_example = False

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if 'id' in attrs:
                self.ids.append(attrs['id'])
            if attrs.get('href', '').startswith('#'):
                self.fragments.append(attrs['href'][1:])
            if tag == 'pre' and attrs.get('aria-label') == 'HYMotion inference example':
                self.in_example = True

        def handle_endtag(self, tag):
            if tag == 'pre':
                self.in_example = False

        def handle_data(self, data):
            if self.in_example:
                self.example.append(data)

    document = Document()
    document.feed((ROOT / 'site/index.html').read_text(encoding='utf-8'))
    assert len(document.ids) == len(set(document.ids))
    assert set(document.fragments) <= set(document.ids)
    example = ''.join(document.example)
    assert 'pipe.infer_text_to_motion' in example
    ast.parse(example)

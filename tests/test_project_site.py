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
                'media/autorig.jpg','media/characters.jpg','media/representations.jpg'}
    assert {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob('*') if p.is_file()} == expected
    class Links(HTMLParser):
        def handle_starttag(self, tag, attrs):
            for key, value in attrs:
                if key in {'src','href','poster'} and value and not urlsplit(value).scheme and not value.startswith('#'):
                    assert (tmp_path / value).exists(), value
    Links().feed((tmp_path / 'index.html').read_text(encoding='utf-8'))


def test_site_build_never_deletes_existing_files(tmp_path):
    existing = tmp_path / 'user-file.txt'
    existing.write_text('preserve')
    with pytest.raises(ValueError, match='must be empty'):
        module.build(tmp_path)
    assert existing.read_text() == 'preserve'

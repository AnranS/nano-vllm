#!/usr/bin/env python3
"""Validate offline HTML links, source snapshots, and function line references."""
import argparse
import ast
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parent


class Document(HTMLParser):
    def __init__(self, path):
        super().__init__(convert_charrefs=True)
        self.ids = set()
        self.urls = []
        self.title = False
        self.main = False
        self.lang = None
        self.duplicates = []
        self.feed(path.read_text())

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            if attrs['id'] in self.ids:
                self.duplicates.append(attrs['id'])
            self.ids.add(attrs['id'])
        if tag == 'html':
            self.lang = attrs.get('lang')
        self.title |= tag == 'title'
        self.main |= tag == 'main'
        for key in ('href', 'src'):
            if key in attrs:
                self.urls.append(attrs[key])


def validate(repo):
    docs = {p.resolve(): Document(p) for p in ROOT.rglob('*.html')}
    links = 0
    for path, doc in docs.items():
        assert not doc.duplicates, (path.name, doc.duplicates)
        assert doc.title and doc.main and doc.lang == 'zh-CN', path
        for url in doc.urls:
            parsed = urlsplit(url)
            assert not parsed.scheme and not parsed.netloc, f'Unexpected external asset/link: {url}'
            target = (path.parent / unquote(parsed.path)).resolve() if parsed.path else path
            assert target.is_relative_to(ROOT), f'Link escapes documentation: {url}'
            assert target.is_file(), f'{path.name}: missing {url}'
            if parsed.fragment:
                assert target in docs and unquote(parsed.fragment) in docs[target].ids, f'{path.name}: missing anchor {url}'
            links += 1
    manifest = json.loads((ROOT / 'source/manifest.json').read_text())
    for name, info in manifest['files'].items():
        source = (repo / name).read_text()
        assert hashlib.sha256(source.encode()).hexdigest() == info['sha256'], f'Stale snapshot: {name}'
        assert len(source.splitlines()) == info['lines'], name
    for ref in manifest['references']:
        if not ref['symbol']:
            continue
        node = ast.parse((repo / ref['path']).read_text())
        for part in ref['symbol'].split('.'):
            node = next(n for n in node.body if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == part)
        assert node.lineno == ref['line'], ref
    print(f'PASS: {len(docs)} HTML pages, {links} local links, {len(manifest["references"])} source references, {len(manifest["files"])} snapshot hashes.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, default=ROOT.parents[1])
    args = parser.parse_args()
    validate(args.repo.resolve())

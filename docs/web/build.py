#!/usr/bin/env python3
"""Build an offline reading guide from the actual local nano-vLLM checkout."""
import argparse
import ast
import hashlib
import html
import json
from pathlib import Path
import subprocess
import tokenize
import io
import keyword

ROOT = Path(__file__).resolve().parent


def esc(value):
    return html.escape(str(value), quote=True)


def source_name(path):
    return path.replace('/', '__').removesuffix('.py') + '.html'


class Context:
    def __init__(self, repo):
        self.repo = repo
        self.commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
        paths = sorted((repo / 'nanovllm').rglob('*.py')) + [repo / 'example.py', repo / 'bench.py']
        self.sources = {str(p.relative_to(repo)): p.read_text() for p in paths}
        self.references = []

    def location(self, path, symbol):
        node = ast.parse(self.sources[path])
        for part in symbol.split('.'):
            node = next(n for n in node.body if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == part)
        return node.lineno, node.end_lineno

    def ref(self, path, symbol=None, label=None):
        line = self.location(path, symbol)[0] if symbol else 1
        self.references.append({'path': path, 'symbol': symbol, 'line': line})
        title = label or (symbol if symbol else path)
        return f'<a class="source-link" href="source/{source_name(path)}#L{line}">{esc(title)} <span>↗ L{line}</span></a>'

    def snippet(self, path, symbol, limit=None):
        start, end = self.location(path, symbol)
        lines = self.sources[path].splitlines()[start - 1:end]
        if limit and len(lines) > limit:
            raise ValueError(f'Choose a shorter source symbol: {symbol}')
        import textwrap
        code = textwrap.dedent('\n'.join(lines))
        return f'<figure class="code-box"><figcaption>{self.ref(path, symbol)}</figcaption><pre><code>{esc(code)}</code></pre></figure>'


def color_lines(source):
    """Syntax color source without changing its text or line numbering."""
    lines = source.splitlines()
    spans = [[] for _ in lines]
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            kind = ('comment' if tok.type == tokenize.COMMENT else 'string' if tok.type == tokenize.STRING else
                    'number' if tok.type == tokenize.NUMBER else 'keyword' if tok.type == tokenize.NAME and keyword.iskeyword(tok.string) else '')
            if not kind:
                continue
            (row, col), (endrow, endcol) = tok.start, tok.end
            for lineno in range(row, min(endrow, len(lines)) + 1):
                spans[lineno - 1].append((col if lineno == row else 0, endcol if lineno == endrow else len(lines[lineno - 1]), kind))
    except tokenize.TokenError:
        pass
    out = []
    for i, line in enumerate(lines):
        cursor, pieces = 0, []
        for start, end, kind in spans[i]:
            pieces.extend([esc(line[cursor:start]), f'<span class="syntax-{kind}">{esc(line[start:end])}</span>'])
            cursor = end
        pieces.append(esc(line[cursor:]))
        out.append(f'<span class="code-line" id="L{i + 1}"><a class="line-number" href="#L{i + 1}" aria-label="第 {i + 1} 行">{i + 1}</a><span class="line-text">{"".join(pieces)}</span></span>')
    return ''.join(out)


def page_html(page, pages, ctx, source=False):
    prefix = '../' if source else ''
    current = page['file']
    links = []
    for i, p in enumerate(pages):
        active = 'aria-current="page"' if p['file'] == current else ''
        links.append(f'<a href="{prefix}{p["file"]}" {active}><span>{i:02}</span>{esc(p["label"])}</a>')
    nav = ''.join(links)
    pager = ''
    if not source:
        idx = next(i for i, p in enumerate(pages) if p['file'] == current)
        previous = f'<a href="{pages[idx-1]["file"]}"><small>上一章</small>← {esc(pages[idx-1]["label"])}</a>' if idx else '<span></span>'
        following = f'<a href="{pages[idx+1]["file"]}"><small>下一章</small>{esc(pages[idx+1]["label"])} →</a>' if idx + 1 < len(pages) else f'<a href="index.html"><small>阅读路线</small>回到目录 →</a>'
        pager = f'<nav class="pager" aria-label="章节翻页">{previous}{following}</nav>'
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light"><meta name="description" content="{esc(page['desc'])}">
<title>{esc(page['title'])} · nano-vLLM 学习手册</title><link rel="stylesheet" href="{prefix}style.css"><script defer src="{prefix}app.js"></script></head>
<body><a class="skip-link" href="#main">跳到正文</a>
<div class="book"><aside class="sidebar"><a class="brand" href="{prefix}index.html"><span class="brand-mark">n.</span><span>nano-vLLM<small>源码学习手册</small></span></a>
<div class="edition">本地源码 · {ctx.commit[:7]}</div><nav class="chapters" aria-label="学习章节">{nav}</nav>
<div class="sidebar-note">先读通一个请求，<br>再理解每一项优化。</div></aside>
<main id="main" class="{'source-main' if source else ''}"><header class="page-header"><span class="eyebrow">{'SOURCE SNAPSHOT / 源码对照' if source else 'NANO-VLLM / 从请求到 GPU'}</span><h1>{esc(page['title'])}</h1><p class="lead">{esc(page['desc'])}</p></header>
<article>{page['body']}</article>{pager}<footer>依据本地工作区生成 · {ctx.commit[:12]} · <a href="{prefix}source/manifest.json">源码版本清单</a> · <a href="{prefix}source/LICENSE.txt">源码 MIT 许可</a></footer></main></div></body></html>'''


def build(repo):
    from content import chapters
    ctx = Context(repo)
    pages = chapters(ctx)
    out = ROOT
    (out / 'source').mkdir(exist_ok=True)
    for p in pages:
        (out / p['file']).write_text(page_html(p, pages, ctx))
    manifest = {'commit': ctx.commit, 'description': 'Local source snapshots, not live file views. Rebuild after editing code.', 'files': {}}
    for path, source in ctx.sources.items():
        manifest['files'][path] = {'sha256': hashlib.sha256(source.encode()).hexdigest(), 'lines': len(source.splitlines())}
        body = f'<p class="note">这是生成文档时的本地源码快照。原文件：<code>{esc(path)}</code>。修改源码后需重新生成。点击行号可定位。</p><div class="source-scroll"><pre class="numbered-source"><code>{color_lines(source)}</code></pre></div>'
        page = {'file': source_name(path), 'title': path, 'desc': '行号与本地文件一致；浏览器离线即可阅读。', 'body': body}
        (out / 'source' / page['file']).write_text(page_html(page, pages, ctx, source=True))
    manifest['references'] = ctx.references
    (out / 'source' / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    (out / 'source' / 'LICENSE.txt').write_text((repo / 'LICENSE').read_text())
    print(f'Built {len(pages)} chapters and {len(ctx.sources)} source snapshots against {ctx.commit[:12]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, default=ROOT.parents[1])
    args = parser.parse_args()
    build(args.repo.resolve())

#!/usr/bin/env python3
"""Serve the local learning guide using only Python's standard library."""
import argparse
import errno
from functools import partial
import hashlib
from http.client import HTTPConnection, HTTPException
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import webbrowser

DOCS_DIR = Path(__file__).resolve().parent
HOST = '127.0.0.1'
HEADER = 'X-NanoVLLM-Docs'
# Distinguish this document directory from another server or another checkout.
DOCS_ID = hashlib.sha256(str(DOCS_DIR).encode()).hexdigest()


class DocsHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header(HEADER, DOCS_ID)
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()


def is_same_server(port):
    connection = HTTPConnection(HOST, port, timeout=2)
    try:
        connection.request('HEAD', '/index.html')
        response = connection.getresponse()
        return response.status == 200 and response.getheader(HEADER) == DOCS_ID
    except (OSError, HTTPException):
        # A port can be occupied by a non-HTTP service. Never stop that process.
        return False
    finally:
        connection.close()


def open_page(url, enabled):
    if enabled:
        try:
            if not webbrowser.open(url):
                print('浏览器未自动打开，请复制上面的地址。', flush=True)
        except (OSError, webbrowser.Error):
            print('浏览器未自动打开，请复制上面的地址。', flush=True)


def main():
    parser = argparse.ArgumentParser(description='启动 nano-vLLM 学习文档的本地服务。')
    parser.add_argument('--port', type=int, default=8769, help='本机端口，默认 8769')
    parser.add_argument('--no-browser', action='store_true', help='启动后不自动打开浏览器')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('--port 必须在 1—65535 之间')
    if not (DOCS_DIR / 'index.html').is_file():
        parser.error('没有找到 index.html，请先运行 docs/web/build.py 生成文档')

    url = f'http://{HOST}:{args.port}/index.html'
    handler = partial(DocsHandler, directory=str(DOCS_DIR))
    try:
        server = ThreadingHTTPServer((HOST, args.port), handler)
    except OSError as error:
        if error.errno == errno.EADDRINUSE:
            if is_same_server(args.port):
                print(f'学习文档服务已在运行，直接复用：\n{url}', flush=True)
                open_page(url, not args.no_browser)
                return 0
            print(f'端口 {args.port} 已被其他服务占用。\n'
                  f'请用 --port 8770 等其他端口启动；原服务保持不变。', file=sys.stderr)
        else:
            print(f'本地服务启动失败：{error}', file=sys.stderr)
        return 1

    with server:
        print(f'nano-vLLM 学习文档已启动：\n{url}\n'
              '仅供本机访问。保持此终端运行，按 Ctrl+C 停止服务。', flush=True)
        try:
            open_page(url, not args.no_browser)
            server.serve_forever()
        except KeyboardInterrupt:
            print('\n学习文档服务已停止。', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())

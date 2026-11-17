#!/usr/bin/env python3
"""Serve only RLT/results as static files; no model, CUDA or training imports.

Example: python scripts/serve_optimization_results.py --port 7860
Open http://127.0.0.1:7860/pilot_expanded_validation_20260908/
The /results/ prefix is also accepted for existing report links.
"""
import argparse
import html
import mimetypes
import re
from email.utils import formatdate
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit


RESULTS_ROOT = Path(__file__).resolve().parents[1] / 'results'


class ResultsHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    server_version = 'RLTStatic/1.0'

    def do_HEAD(self):
        self.serve(head=True)

    def do_GET(self):
        self.serve(head=False)

    def resolve_request(self):
        raw = unquote(urlsplit(self.path).path, errors='strict')
        if '\\' in raw or '\x00' in raw or any(p in ('.', '..') for p in raw.split('/')):
            raise ValueError('不允许路径穿越')
        relative = raw.lstrip('/')
        if relative == 'results' or relative.startswith('results/'):
            relative = relative[len('results'):].lstrip('/')
        root = RESULTS_ROOT.resolve()
        target = (root / relative).resolve()
        if not target.is_relative_to(root):
            raise ValueError('仅允许读取 RLT/results 内的文件')
        return target

    def send_file_headers(self, code, size, kind, *, modified=None, content_range=None):
        self.send_response(code)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(size))
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('X-Content-Type-Options', 'nosniff')
        if modified is not None:
            self.send_header('Last-Modified', formatdate(modified, usegmt=True))
        if content_range:
            self.send_header('Content-Range', content_range)
        self.end_headers()

    def listing(self, target, head):
        items = []
        for child in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            if not child.resolve().is_relative_to(RESULTS_ROOT.resolve()):
                continue
            suffix = '/' if child.is_dir() else ''
            items.append('<li><a href="' + quote(child.name, safe='') + suffix + '">'
                         + html.escape(child.name + suffix) + '</a></li>')
        body = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width,initial-scale=1">'
                '<title>RLT 评估结果</title><body style="font:16px/1.8 system-ui;margin:32px">'
                '<h1>RLT 评估结果</h1><p>只读静态文件；打开评估目录中的 index.html。</p><ul>'
                + ''.join(items) + '</ul></body></html>').encode('utf-8')
        self.send_file_headers(200, len(body), 'text/html; charset=utf-8')
        if not head:
            self.wfile.write(body)

    def serve(self, head):
        try:
            target = self.resolve_request()
        except (ValueError, UnicodeError):
            self.send_error(403, 'Path is outside the results directory or invalid')
            return
        try:
            if target.is_dir():
                request = urlsplit(self.path)
                if not request.path.endswith('/'):
                    self.send_response(301)
                    # Use a local absolute path, never a scheme-relative Location.
                    location = '/' + request.path.lstrip('/') + '/'
                    if request.query:
                        location += '?' + request.query
                    self.send_header('Location', location)
                    self.send_header('Content-Length', '0')
                    self.end_headers()
                    return
                index = (target / 'index.html').resolve()
                if not index.is_relative_to(RESULTS_ROOT.resolve()):
                    self.send_error(403, 'Index is outside the results directory')
                    return
                if index.is_file():
                    target = index
                else:
                    self.listing(target, head)
                    return
            if not target.is_file():
                self.send_error(404, 'File not found')
                return
            with target.open('rb') as handle:
                import os
                stat = os.fstat(handle.fileno())
                size = stat.st_size
                start, end, code = 0, size - 1, 200
                requested = self.headers_range()
                if requested:
                    match = re.fullmatch(r'bytes=(\d*)-(\d*)', requested.strip())
                    valid = bool(match and any(match.groups()) and size)
                    if valid:
                        first, last = match.groups()
                        if first:
                            start = int(first)
                            end = min(int(last), size - 1) if last else size - 1
                        else:
                            length = int(last)
                            valid = length > 0
                            start, end = max(0, size - length), size - 1
                        valid = valid and 0 <= start <= end < size
                    if not valid:
                        self.send_file_headers(416, 0, 'text/plain', content_range=f'bytes */{size}')
                        return
                    code = 206
                count = max(0, end - start + 1)
                content_type = mimetypes.guess_type(target.name)[0] or 'application/octet-stream'
                if content_type.startswith('text/') or content_type == 'application/json':
                    content_type += '; charset=utf-8'
                self.send_file_headers(code, count, content_type, modified=stat.st_mtime,
                             content_range=f'bytes {start}-{end}/{size}' if code == 206 else None)
                if not head:
                    handle.seek(start)
                    remaining = count
                    while remaining:
                        chunk = handle.read(min(256 * 1024, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except (OSError, ValueError):
            self.send_error(404, 'File unavailable')

    def headers_range(self):
        # BaseHTTPRequestHandler stores incoming HTTP headers on self.headers.
        return self.headers.get('Range')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()
    if not RESULTS_ROOT.is_dir():
        parser.error(f'Results directory does not exist: {RESULTS_ROOT}')
    try:
        server = ThreadingHTTPServer((args.host, args.port), ResultsHandler)
    except OSError as exc:
        parser.exit(1, f'无法监听 {args.host}:{args.port}：{exc}；不会停止已有服务。\n')
    print(f'只读结果服务：http://{args.host}:{server.server_port}/  根目录：{RESULTS_ROOT}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Download a pinned public OpenPI checkpoint inventory, verifying GCS CRC32C."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request

import google_crc32c

ROOT = Path(__file__).resolve().parents[1]

def verify(path, item):
    if path.stat().st_size != int(item['size']):
        raise ValueError(f'Size mismatch: {path}')
    crc = google_crc32c.Checksum()
    sha = hashlib.sha256()
    with path.open('rb') as f:
        while block := f.read(4 * 1024 * 1024):
            crc.update(block)
            sha.update(block)
    if base64.b64encode(crc.digest()).decode() != item['crc32c']:
        raise ValueError(f'CRC32C mismatch: {path}; file retained for inspection')
    return sha.hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name', choices=['pi05_libero', 'pi05_base'], required=True)
    p.add_argument('--workers', type=int, default=3)
    args = p.parse_args()
    if not 1 <= args.workers <= 8:
        p.error('workers must be between 1 and 8')
    manifest = ROOT / 'manifests' / f'{args.name}_remote_inventory.json'
    data = json.loads(manifest.read_text())
    if data.get('nextPageToken') or not data.get('items'):
        raise ValueError('Inventory must be complete and nonempty')
    dest = ROOT / 'checkpoints' / args.name
    dest.mkdir(parents=True, exist_ok=True)
    prefix = f'checkpoints/{args.name}/'
    def fetch(item):
        if not item['name'].startswith(prefix):
            raise ValueError('Object outside checkpoint prefix')
        relative = Path(item['name'][len(prefix):])
        target = (dest / relative).resolve()
        if not target.is_relative_to(dest.resolve()):
            raise ValueError('Unsafe object path')
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            digest = verify(target, item)
        else:
            partial = target.with_name(target.name + '.download-partial')
            size = int(item['size'])
            url = ('https://storage.googleapis.com/openpi-assets/'
                   + urllib.parse.quote(item['name'], safe='/')
                   + '?generation=' + item['generation'])
            for attempt in range(4):
                start = partial.stat().st_size if partial.exists() else 0
                if start == size:
                    break
                if start > size:
                    raise ValueError(f'Oversized partial: {partial}')
                try:
                    req = urllib.request.Request(url, headers={'Range': f'bytes={start}-'} if start else {})
                    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                    with opener.open(req, timeout=45) as response:
                        if start and (response.status != 206 or not response.headers.get('Content-Range','').startswith(f'bytes {start}-')):
                            raise ValueError('Resume Range was not honored; refusing to append')
                        last = time.monotonic()
                        with partial.open('ab' if start else 'wb') as f:
                            while block := response.read(1024 * 1024):
                                f.write(block)
                                if time.monotonic()-last > 20:
                                    print(json.dumps(dict(file=str(relative),bytes=f.tell(),total=size)), flush=True)
                                    last = time.monotonic()
                    break
                except (OSError, TimeoutError) as exc:
                    print(json.dumps(dict(file=str(relative),attempt=attempt+1,error=str(exc))),flush=True)
                    if attempt == 3:
                        raise
            digest = verify(partial,item)
            partial.replace(target)
        row = dict(path=str(target),size=int(item['size']),generation=item['generation'],crc32c=item['crc32c'],sha256=digest)
        print(json.dumps(dict(verified=str(relative),bytes=row['size'])),flush=True)
        return row
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        files = list(pool.map(fetch,data['items']))
    report = dict(status='complete',checkpoint=args.name,source='gs://openpi-assets/checkpoints/'+args.name,
                  total_bytes=sum(f['size'] for f in files),files=files)
    (ROOT/'manifests'/f'{args.name}_download_verified.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='complete',checkpoint=args.name,bytes=report['total_bytes'])),flush=True)

if __name__ == '__main__':
    main()

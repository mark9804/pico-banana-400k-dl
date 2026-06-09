import os
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import urllib.parse
from functools import partial
import json
import threading
import time
import urllib.error

IDX_PATH = Path(__file__).parent / "manifest"
RESOURCE_FILE = IDX_PATH / "multi_turn_manifest.txt"
JSONL_FILE = IDX_PATH / "multi-turn.jsonl"
IMAGES_BASE_PATH = Path(__file__).parent / "images"
TARGET_PATH = IMAGES_BASE_PATH / "multi-turn"
ERROR_FILE = Path(__file__).parent / "error.jsonl"
MAX_RETRIES = 3

HOST_NAME = "ml-site.cdn-apple.com" # can be changed for reverse proxy
MULTI_TURN_MANIFEST_URL = "https://ml-site.cdn-apple.com/datasets/pico-banana-300k/nb/manifest/multi_turn_manifest.txt"
MULTI_TURN_INSTRUCTION_URL = "https://ml-site.cdn-apple.com/datasets/pico-banana-300k/nb/jsonl/multi-turn.jsonl"

error_file_lock = threading.Lock()


SKIPPED_URLS = set()
LOGGED_URLS = set()

if ERROR_FILE.exists():
    with open(ERROR_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                url = data.get("url")
                if not url:
                    continue
                LOGGED_URLS.add(url)
                if data.get("code") in (404, 410):
                    SKIPPED_URLS.add(url)
            except json.JSONDecodeError:
                pass

def log_error(url, error_obj):
    with error_file_lock:
        if url in LOGGED_URLS:
            return
            
        code = getattr(error_obj, "code", None)
        reason = getattr(error_obj, "reason", str(error_obj))
        
        # URLError's reason can be an exception itself
        if isinstance(reason, Exception):
            reason = str(reason)
            
        error_data = {"url": url, "reason": reason}
        if code is not None:
            error_data["code"] = code
            
        with open(ERROR_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(error_data) + "\n")
            
        LOGGED_URLS.add(url)
        if code in (404, 410):
            SKIPPED_URLS.add(url)

def download_file(url: str, target_file: Path, pbar_bytes=None):
    if target_file.exists():
        return
        
    if url in SKIPPED_URLS:
        return

    tmp_file = target_file.parent / f"{target_file.name}.tmp"
    attempt = 0
    non_429_failures = 0
    while non_429_failures < MAX_RETRIES:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as response, open(tmp_file, 'wb') as out_file:
                while True:
                    chunk = response.read(32768)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    if pbar_bytes is not None:
                        pbar_bytes.update(len(chunk))
            
            os.replace(tmp_file, target_file)
            return
        except urllib.error.HTTPError as e:
            if tmp_file.exists():
                try:
                    os.remove(tmp_file)
                except OSError:
                    pass
            
            if e.code in (404, 410):
                log_error(url, e)
                return
            elif e.code == 429:
                # retry until non-429 error happens
                backoff = 2 ** attempt
                tqdm.write(f"HTTP 429 Too Many Requests for {url}. Backing off for {backoff} seconds...")
                time.sleep(backoff)
                attempt += 1
            else:
                non_429_failures += 1
                if non_429_failures == MAX_RETRIES:
                    tqdm.write(f"Failed after {MAX_RETRIES} attempts: {url} - {e}")
                    log_error(url, e)
                else:
                    time.sleep(1)
        except Exception as e:
            if tmp_file.exists():
                try:
                    os.remove(tmp_file)
                except OSError:
                    pass
            
            non_429_failures += 1
            if non_429_failures == MAX_RETRIES:
                tqdm.write(f"Failed after {MAX_RETRIES} attempts: {url} - {e}")
                log_error(url, e)
            else:
                time.sleep(1)


def download_image(url, pbar_bytes=None):
    if not url:
        return

    parsed_url = urllib.parse.urlparse(url)
    if HOST_NAME and parsed_url.netloc != HOST_NAME:
        url = urllib.parse.urlunparse(parsed_url._replace(netloc=HOST_NAME))

    filename = url.split('/')[-1]
    target_file = TARGET_PATH / filename

    download_file(url, target_file, pbar_bytes)

def main():
    for path in [IDX_PATH, TARGET_PATH]:
        if path.exists():
            continue
        else:
            os.makedirs(path)
    
    for url in [MULTI_TURN_INSTRUCTION_URL, MULTI_TURN_MANIFEST_URL]:
        filename = url.split("/")[-1]
        target_file = IDX_PATH / filename
        if target_file.exists():
            continue
        print(f"Downloading {filename}...")
        download_file(url, target_file)

    def iter_urls():
        with open(RESOURCE_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line

    print(f"Downloading to: {IMAGES_BASE_PATH}")
    
    def get_total_items() -> int:
        with open(RESOURCE_FILE, "r") as f:
            return sum(1 for line in f if line.strip())
            
    total = get_total_items()
    
    with ThreadPoolExecutor(max_workers=16) as executor:
        with tqdm(total=total, desc="Files", position=0, unit="file") as pbar, \
             tqdm(desc="Downloaded", unit="B", unit_scale=True, unit_divisor=1024, position=1) as pbar_bytes:
            
            func = partial(download_image, pbar_bytes=pbar_bytes)
            for _ in executor.map(func, iter_urls()):
                pbar.update(1)

    print("Download completed.")

if __name__ == "__main__":
    main()

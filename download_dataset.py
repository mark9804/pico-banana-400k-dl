import os
import urllib.request
import shutil
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
rate_limit_lock = threading.Lock()
last_external_request_time = 0.0
EXTERNAL_REQUEST_INTERVAL = 0.5 # minimum wait time between external requests

def log_error(url, reason):
    with error_file_lock:
        with open(ERROR_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps({"url": url, "reason": str(reason)}) + "\n")

def download_file(url: str, target_file: Path, is_external: bool = False, pbar_bytes=None):
    if target_file.exists():
        return

    tmp_file = target_file.parent / f"{target_file.name}.tmp"
    attempt = 0
    non_429_failures = 0
    while non_429_failures < MAX_RETRIES:
        if is_external:
            with rate_limit_lock:
                global last_external_request_time
                now = time.time()
                elapsed = now - last_external_request_time
                if elapsed < EXTERNAL_REQUEST_INTERVAL:
                    time.sleep(EXTERNAL_REQUEST_INTERVAL - elapsed)
                last_external_request_time = time.time()

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
            
            if e.code == 429:
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


def download_image(item, pbar_bytes=None):
    url, is_external = item
    if not url:
        return

    parsed_url = urllib.parse.urlparse(url)
    
    if is_external:
        # https://farm8.staticflickr.com/2915/14573719235_6cfb811e3c_o.jpg
        # host: farm8.staticflickr.com, path: /2915/14573719235_6cfb811e3c_o.jpg
        host_name = parsed_url.netloc
        path_name = parsed_url.path.strip('/') # 2915/14573719235_6cfb811e3c_o.jpg
        
        target_file = IMAGES_BASE_PATH / host_name / path_name
        target_dir = target_file.parent
        target_dir.mkdir(parents=True, exist_ok=True)
    else:
        if HOST_NAME and parsed_url.netloc != HOST_NAME:
            url = urllib.parse.urlunparse(parsed_url._replace(netloc=HOST_NAME))
    
        filename = url.split('/')[-1]
        target_file = TARGET_PATH / filename

    download_file(url, target_file, is_external, pbar_bytes)

def main():
    for path in [IDX_PATH, TARGET_PATH]:
        if path.exists():
            continue
        else:
            os.makedirs(path)
    
    for url in [MULTI_TURN_INSTRUCTION_URL, MULTI_TURN_MANIFEST_URL]:
        filename = url.split("/")[-1]
        target_file = IDX_PATH / filename
        print(f"Downloading {filename}...")
        download_file(url, target_file, is_external=False)

    def iter_urls():
        # yield internal manifest URLs
        with open(RESOURCE_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield (line, False)
                    
        # yield external URLs from jsonl
        external_urls = set()
        with open(JSONL_FILE, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                content = json.loads(line)
                for file_obj in content.get("files", []):
                    url = file_obj.get("url", "")
                    if url.startswith("http://") or url.startswith("https://"):
                        if url not in external_urls:
                            external_urls.add(url)
                            yield (url, True)

    print(f"Downloading to: {IMAGES_BASE_PATH}")
    
    def get_total_items() -> int:
        count = 0
        with open(RESOURCE_FILE, "r") as f:
            count += sum(1 for line in f if line.strip())
            
        external_urls = set()
        with open(JSONL_FILE, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                content = json.loads(line)
                for file_obj in content.get("files", []):
                    url = file_obj.get("url", "")
                    if url.startswith("http://") or url.startswith("https://"):
                        external_urls.add(url)
        count += len(external_urls)
        return count
            
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

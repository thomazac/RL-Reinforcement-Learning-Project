"""Download ranked-doubles replays from ballchasing.com into data/replays."""

import argparse
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

API = "https://ballchasing.com/api"


def get(session, url, **kw):
    while True:
        r = session.get(url, timeout=60, **kw)
        if r.status_code == 429:
            time.sleep(float(r.headers.get("Retry-After", 5)))
            continue
        r.raise_for_status()
        return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--min-rank", default="grand-champion-1")
    ap.add_argument("--max-rank", default="grand-champion-1")
    ap.add_argument("--out", default="data/replays")
    args = ap.parse_args()

    load_dotenv()
    session = requests.Session()
    session.headers["Authorization"] = os.environ["BALLCHASING_API_KEY"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    params = {"playlist": "ranked-doubles", "min-rank": args.min_rank, "max-rank": args.max_rank,
              "count": min(args.count, 200), "sort-by": "replay-date", "sort-dir": "desc"}
    url, downloaded = f"{API}/replays", 0
    while url and downloaded < args.count:
        page = get(session, url, params=params).json()
        params = None  # the "next" link already carries the query
        for replay in page.get("list", []):
            path = out / f"{replay['id']}.replay"
            if not path.exists():
                path.write_bytes(get(session, f"{API}/replays/{replay['id']}/file").content)
            downloaded += 1
            print(f"{downloaded}/{args.count} {path.name}", flush=True)
            if downloaded >= args.count:
                break
        url = page.get("next")


if __name__ == "__main__":
    main()

# Download and unzip a MovieLens dataset

from __future__ import annotations
import argparse
import ssl
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        try:
            return ssl.create_default_context()
        except Exception:
            return ssl._create_unverified_context()

DATA_DIR = Path(__file__).resolve().parent

DATASETS = {
    "full": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
        "folder": "ml-25m",
    },
    "small": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
        "folder": "ml-latest-small",
    },
}


def _download(url: str, dest: Path) -> None:
    """Stream a URL to disk with a simple progress readout."""
    print(f"Downloading {url}")
    with urlopen(url, context=_ssl_context()) as resp:  # noqa: S310 (trusted host)
        total = int(resp.headers.get("Content-Length", 0))
        read = 0
        chunk = 1 << 20  # 1 MiB
        with open(dest, "wb") as f:
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                f.write(buf)
                read += len(buf)
                if total:
                    pct = 100 * read / total
                    print(f"\r  {read / 1e6:7.1f} / {total / 1e6:7.1f} MB "
                          f"({pct:5.1f}%)", end="", flush=True)
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--small", action="store_true",
                    help="Use ml-latest-small instead of ml-25m.")
    args = ap.parse_args()

    spec = DATASETS["small" if args.small else "full"]
    folder = DATA_DIR / spec["folder"]
    if (folder / "ratings.csv").exists():
        print(f"Already present: {folder}")
        return 0

    zip_path = DATA_DIR / f"{spec['folder']}.zip"
    if not zip_path.exists():
        _download(spec["url"], zip_path)

    print(f"Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(DATA_DIR)
    zip_path.unlink(missing_ok=True)
    print(f"Done. Data at {folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

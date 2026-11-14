"""Download a small subset of a Hugging Face dataset without the `hf` CLI.

The project venv already has `requests`, so this avoids installing
`huggingface_hub` (which is currently blocked by PyPI connectivity).

Gated datasets need a Hugging Face read token. Create one at:
https://huggingface.co/settings/tokens

Usage:
    $env:HF_TOKEN = "hf_..."
    .\\.venv\\Scripts\\python.exe scripts\\download_hf_subset.py --shards 2

Downloads:
    annotations/detection.json
    annotations/scene_attributes.json
    data/train-00000-of-00646.tar  (up to --shards)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

import requests

DEFAULT_REPO = "thirdeyelabs/indian-road-dataset"
DEFAULT_DEST = Path("data/raw/public/indian-road")
SHARD_TEMPLATE = "data/train-{index:05d}-of-00646.tar"
ANNOTATION_FILES = [
    "annotations/detection.json",
    "annotations/scene_attributes.json",
]
CHUNK_SIZE = 1024 * 1024  # 1 MiB


def build_file_list(shards: int, extra_files: Iterable[str] | None) -> list[str]:
    if extra_files:
        files = list(extra_files)
    else:
        files = list(ANNOTATION_FILES)
    files.extend(SHARD_TEMPLATE.format(index=index) for index in range(shards))
    # Keep order, drop duplicates.
    return list(dict.fromkeys(files))


def human_size(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024 or unit == "GB":
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} GB"


def download_file(
    session: requests.Session,
    repo: str,
    repo_path: str,
    destination: Path,
    token: str | None,
    overwrite: bool,
) -> bool:
    if destination.exists() and destination.stat().st_size > 0 and not overwrite:
        print(f"skip (exists): {repo_path} ({human_size(destination.stat().st_size)})")
        return True

    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{repo_path}?download=true"
    headers = {"User-Agent": "road-safety-analytics/0.1"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")

    try:
        with session.get(
            url,
            headers=headers,
            stream=True,
            timeout=(30, 120),
            allow_redirects=True,
        ) as response:
            if response.status_code in (401, 403):
                print(
                    f"ERROR {repo_path}: {response.status_code} - token missing, "
                    "invalid, or dataset access not accepted.",
                    file=sys.stderr,
                )
                print(
                    "Open the dataset page, accept the gate, then create a read "
                    "token at https://huggingface.co/settings/tokens",
                    file=sys.stderr,
                )
                return False
            if response.status_code == 404:
                print(f"ERROR {repo_path}: 404 - file does not exist.", file=sys.stderr)
                return False
            response.raise_for_status()

            total = int(response.headers.get("Content-Length") or 0)
            downloaded = 0
            next_report = 0
            print(f"download: {repo_path}" + (f" ({human_size(total)})" if total else ""))

            with temporary.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    downloaded += len(chunk)
                    if downloaded >= next_report:
                        if total:
                            percent = downloaded * 100 / total
                            print(
                                f"  {human_size(downloaded)} / {human_size(total)} "
                                f"({percent:.1f}%)"
                            )
                            next_report = downloaded + max(total // 10, CHUNK_SIZE)
                        else:
                            print(f"  {human_size(downloaded)}")
                            next_report = downloaded + 100 * CHUNK_SIZE

        temporary.replace(destination)
        print(f"saved: {destination} ({human_size(downloaded)})")
        return True
    except requests.RequestException as exc:
        print(f"ERROR {repo_path}: {exc}", file=sys.stderr)
        return False
    finally:
        if temporary.exists() and temporary.stat().st_size == 0:
            temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download selected files from a Hugging Face dataset."
    )
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument(
        "--shards",
        type=int,
        default=2,
        help="Number of train tar shards to download, starting at 00000. Default: 2.",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Extra repo-relative files to download.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        help="HF read token; defaults to HF_TOKEN.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.shards < 0:
        print("--shards must be >= 0", file=sys.stderr)
        return 2

    files = build_file_list(args.shards, args.files)
    token = args.token.strip() if args.token else None
    if not token:
        print(
            "No HF_TOKEN set. Gated files will return 401; public files may work.",
            file=sys.stderr,
        )

    session = requests.Session()
    failures = 0
    for repo_path in files:
        destination = args.dest / repo_path
        if not download_file(
            session,
            repo=args.repo,
            repo_path=repo_path,
            destination=destination,
            token=token,
            overwrite=args.overwrite,
        ):
            failures += 1

    print(f"\nDone: {len(files) - failures}/{len(files)} files in {args.dest}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Copy the page and its icons into a static site folder.

Standard library only, so a page-only deploy needs no weather or model packages.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
HTML_TAG = '<html lang="sv">'
# Not named "api": Vercel treats a root api/ folder as server functions.
API_DIR = "api-data"


def static_page(html: str) -> str:
    """Mark the page as static, so it reads the JSON files instead of calling the app server."""
    if html.count(HTML_TAG) != 1:
        raise ValueError(f"Expected exactly one {HTML_TAG} in index.html")
    return html.replace(HTML_TAG, '<html lang="sv" data-static-api="1">')


def write_page(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    page = static_page((ROOT_DIR / "index.html").read_text(encoding="utf-8"))
    (out / "index.html").write_text(page, encoding="utf-8")
    shutil.copytree(ROOT_DIR / "icons", out / "icons", dirs_exist_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT_DIR / "site", help="Output folder")
    write_page(parser.parse_args().out)

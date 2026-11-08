"""Capture the README screenshots with headless Chrome. Start the server first: make run

uv run python scripts/screenshot.py --port 8828
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

CHROME_PATHS = ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "google-chrome", "chromium"]
DOCS = Path(__file__).resolve().parents[1] / "docs"


def chrome() -> str:
    for c in CHROME_PATHS:
        if Path(c).exists() or shutil.which(c):
            return c
    raise SystemExit("Chrome not found")


def shoot(url: str, out: Path, width: int, height: int, dark: bool = False) -> None:
    args = [chrome(), "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={width},{height}",
            "--virtual-time-budget=8000", f"--screenshot={out}"]
    args.append("--force-dark-mode" if dark else "--blink-settings=preferredColorScheme=1")
    subprocess.run(args + [url], check=True, capture_output=True, timeout=60)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8828)
    a = ap.parse_args()
    base = f"http://127.0.0.1:{a.port}"
    DOCS.mkdir(exist_ok=True)
    shoot(f"{base}/?sample=msa-04", DOCS / "review.png", 1440, 1000)
    shoot(f"{base}/?sample=msa-04", DOCS / "review-dark.png", 1440, 1000, dark=True)


if __name__ == "__main__":
    main()

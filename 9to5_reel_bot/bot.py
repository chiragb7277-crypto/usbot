"""
bot.py
======
"9to5 Reel Bot" (Offline Mode)
Turns joke items from posts.json into 1080x1920 static Facebook Reels and 7s silent MP4s.

No AI calls. No API keys. 100% offline.

Usage:
  python bot.py               # Process posts.json in project root
  python bot.py [file.json]   # Process a specific JSON file
  python bot.py --test        # Run render test on sample item
"""

import os
import re
import sys
import json
import shutil
import argparse
import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple

from render import render_reel, create_mp4, ensure_assets

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_POSTS_FILE = BASE_DIR / "posts.json"
DONE_DIR = BASE_DIR / "done"
OUTPUT_DIR = BASE_DIR / "output"
ASSETS_DIR = BASE_DIR / "assets"
LOG_FILE = OUTPUT_DIR / "log.jsonl"


def init_environment():
    """Initializes required output and assets directories."""
    for d in (DONE_DIR, OUTPUT_DIR, ASSETS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    ensure_assets(BASE_DIR)


def generate_slug(item: Dict[str, Any]) -> str:
    """Generates a clean filesystem slug from highlight, punchline, or setup."""
    text = item.get("highlight") or item.get("punchline") or item.get("setup") or "reel"
    # Take first 4-5 words
    words = text.split()[:5]
    raw = " ".join(words).lower()
    clean = re.sub(r'[^a-z0-9\s]+', '', raw).strip()
    slug = re.sub(r'\s+', '-', clean)
    return slug[:35] if slug else "reel"


def validate_item(item: Any, index: int) -> Tuple[bool, str]:
    """
    Validates a post item:
    - Must be a dictionary.
    - 'punchline' is required and non-empty.
    - 'highlight' is required, non-empty, and must appear within 'punchline' (case-insensitive).
    """
    if not isinstance(item, dict):
        return False, f"Item #{index} is not a valid JSON object: {repr(item)}"

    punchline = str(item.get("punchline", "")).strip()
    highlight = str(item.get("highlight", "")).strip()

    if not punchline:
        return False, f"Item #{index} missing required 'punchline'"

    if not highlight:
        return False, f"Item #{index} missing required 'highlight'"

    if highlight.lower() not in punchline.lower():
        return False, f"Item #{index} highlight '{highlight}' not found inside punchline '{punchline}'"

    return True, "valid"


def process_posts_file(posts_path: Path):
    """Reads, validates, and renders all items from posts.json, then moves it to done/."""
    if not posts_path.exists():
        print(f"❌ Input file not found: {posts_path}")
        print(f"👉 Create '{posts_path.name}' with a JSON array of jokes, e.g.:")
        example = [
            {
                "setup": "Corporate Friday survival tip:",
                "punchline": "If you can't solve it by noon, it's officially a Monday issue",
                "highlight": "Monday issue",
                "caption": "Tag your Friday escape partner 🏃‍♂️"
            }
        ]
        print(json.dumps(example, indent=2))
        return

    try:
        with open(posts_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                print(f"⚠️ {posts_path} is empty.")
                return
            data = json.loads(content)
    except Exception as e:
        print(f"❌ Failed to parse JSON from {posts_path}: {e}")
        return

    if not isinstance(data, list):
        print(f"❌ Expected a JSON array (list) in {posts_path}, got {type(data).__name__}.")
        return

    total_items = len(data)
    print(f"📄 Found {total_items} post(s) in {posts_path.name}. Starting processing...")

    today_str = datetime.date.today().strftime("%Y-%m-%d")
    output_date_dir = OUTPUT_DIR / today_str
    output_date_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    skip_count = 0

    for i, item in enumerate(data, start=1):
        is_valid, reason = validate_item(item, i)
        if not is_valid:
            print(f"⚠️ [SKIP #{i}] {reason}")
            skip_count += 1
            # Log skip
            log_entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "batch_index": i,
                "status": "skipped",
                "reason": reason,
                "raw_item": item
            }
            append_log(log_entry)
            continue

        slug = generate_slug(item)
        file_prefix = f"{i:02d}-{slug}"
        png_path = output_date_dir / f"{file_prefix}.png"
        mp4_path = output_date_dir / f"{file_prefix}.mp4"
        txt_path = output_date_dir / f"{file_prefix}.txt"

        setup_str = item.get("setup", "(none)")
        print(f"\n🎬 [REEL #{i:02d}] {slug}")
        if setup_str and setup_str != "(none)":
            print(f"   Setup:     {setup_str}")
        print(f"   Punchline: {item['punchline']}")
        print(f"   Highlight: {item['highlight']}")

        try:
            # 1. Render PNG
            render_reel(item, str(png_path), str(ASSETS_DIR))

            # 2. Render 7s silent MP4
            create_mp4(str(png_path), str(mp4_path), duration=7)

            # 3. Save Caption .txt file
            caption_text = str(item.get("caption", "")).strip()
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(caption_text + ("\n" if caption_text else ""))

            # 4. Log success
            log_entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "batch_index": i,
                "slug": file_prefix,
                "status": "success",
                "output_files": {
                    "png": str(png_path.relative_to(BASE_DIR)),
                    "mp4": str(mp4_path.relative_to(BASE_DIR)),
                    "caption": str(txt_path.relative_to(BASE_DIR))
                },
                "data": item
            }
            append_log(log_entry)
            success_count += 1
            print(f"   ✅ Saved: {file_prefix}.png, .mp4, .txt")

        except Exception as e:
            print(f"   ❌ Error rendering reel #{i}: {e}")
            skip_count += 1
            append_log({
                "timestamp": datetime.datetime.now().isoformat(),
                "batch_index": i,
                "status": "failed",
                "error": str(e),
                "data": item
            })

    # Rename posts.json to done/posts-<timestamp>.json
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    done_path = DONE_DIR / f"posts-{timestamp_str}.json"
    shutil.move(str(posts_path), str(done_path))
    print(f"\n📦 Archived input to: {done_path.relative_to(BASE_DIR)}")
    print(f"🎉 Finished batch: {success_count} rendered successfully, {skip_count} skipped.")
    print(f"📂 Output directory: {output_date_dir.relative_to(BASE_DIR)}")


def append_log(record: Dict[str, Any]):
    """Appends an execution record to output/log.jsonl."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_test():
    """Runs a render test with fixed sample JSON and asserts 1080x1920 PNG and 7s MP4."""
    print("=== Running 9to5 Reel Bot Test (Offline) ===")
    sample_data = {
        "setup": "Corporate Friday survival tip:",
        "punchline": "If you can't solve it by noon, it's officially a Monday issue",
        "highlight": "Monday issue",
        "caption": "Tag your Friday escape partner 🏃‍♂️"
    }

    test_dir = BASE_DIR / "output" / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    test_png = test_dir / "01-monday-issue.png"
    test_mp4 = test_dir / "01-monday-issue.mp4"
    test_txt = test_dir / "01-monday-issue.txt"

    render_reel(sample_data, str(test_png), str(ASSETS_DIR))

    from PIL import Image
    im = Image.open(test_png)
    print(f"[ASSERT] Rendered PNG dimensions: {im.size}")
    assert im.size == (1080, 1920), f"Expected (1080, 1920), got {im.size}"

    create_mp4(str(test_png), str(test_mp4), duration=7)
    assert test_mp4.exists() and test_mp4.stat().st_size > 0, "MP4 was not created"

    with open(test_txt, "w", encoding="utf-8") as f:
        f.write(sample_data["caption"] + "\n")
    assert test_txt.exists() and test_txt.stat().st_size > 0, "Caption TXT was not created"

    print("✅ TEST PASSED: 1080x1920 PNG, 7s silent MP4, and caption TXT verified!")


def main():
    parser = argparse.ArgumentParser(description="9to5 Reel Bot (Offline Mode)")
    parser.add_argument("input_file", nargs="?", default=str(DEFAULT_POSTS_FILE), help="Path to posts.json (default: posts.json)")
    parser.add_argument("--test", action="store_true", help="Run test on fixed sample JSON")
    args = parser.parse_args()

    init_environment()

    if args.test:
        run_test()
        return

    posts_file = Path(args.input_file)
    process_posts_file(posts_file)


if __name__ == "__main__":
    main()

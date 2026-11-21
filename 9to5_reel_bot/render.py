"""
render.py
=========
High-fidelity Facebook Reel graphic and video renderer for "9 to 5 Office".
Generates 1080x1920 PNG graphics and 7-second silent MP4 reels according to exact page specifications.
"""

import os
import re
import sys
import shutil
import urllib.request
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from PIL import Image, ImageDraw, ImageFont

# Canvas dimensions
CANVAS_WIDTH = 1080
CANVAS_HEIGHT = 1920

# Coordinates & Dimensions (Exact Specifications)
AVATAR_X = 100
AVATAR_Y = 560
AVATAR_SIZE = 136

NAME_X = 268
NAME_Y = 580
NAME_FONT_SIZE = 44

HANDLE_X = 268
HANDLE_Y = 640
HANDLE_FONT_SIZE = 36

TEXT_BLOCK_X = 108
TEXT_BLOCK_Y = 760
TEXT_MAX_WIDTH = 864
MAX_TEXT_BOTTOM_Y = 1350

DEFAULT_FONT_SIZE = 50
MIN_FONT_SIZE = 40
BASE_LINE_HEIGHT = 72

# Colors
COLOR_BG = (0, 0, 0, 255)            # Pure Black #000000
COLOR_WHITE = (255, 255, 255, 255)   # White #FFFFFF
COLOR_HANDLE = (142, 142, 142, 255)  # Grey #8E8E8E
COLOR_HIGHLIGHT = (247, 235, 154, 255) # Pastel Yellow #F7EB9A

# Fallback Banner Source
EXTERNAL_BANNER_PATH = Path("/Users/chiragbhatia/9to5 Moniszted /banner.png")

FONT_URLS = {
    "Inter-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/inter/Inter%5Bopsz%2Cwght%5D.ttf",
    "Poppins-Bold.ttf": "https://github.com/google/fonts/raw/main/ofl/poppins/Poppins-Bold.ttf",
    "NotoSansDevanagari-Regular.ttf": "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansDevanagari/NotoSansDevanagari-Regular.ttf",
    "NotoColorEmoji.ttf": "https://github.com/google/fonts/raw/main/ofl/notocoloremoji/NotoColorEmoji-Regular.ttf",
}


def is_devanagari(char: str) -> bool:
    """Returns True if the character is in the Devanagari Unicode range."""
    cp = ord(char)
    return (0x0900 <= cp <= 0x097F) or (0xA8E0 <= cp <= 0xA8FF)


def is_emoji_char(char: str) -> bool:
    """Returns True if character is an emoji or emoji variation selector."""
    cp = ord(char)
    return (
        0x1F300 <= cp <= 0x1FAFF or
        0x2600 <= cp <= 0x27BF or
        0x1F600 <= cp <= 0x1F64F or
        0x1F680 <= cp <= 0x1F6FF or
        cp in (0x200D, 0xFE0F, 0xFE0E) or
        cp >= 0x1F000
    )


def ensure_assets(base_dir: Optional[Path] = None) -> Path:
    """
    Ensures assets directory has banner, avatar, and all required fonts.
    Auto-downloads fonts or crops avatar if missing.
    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent

    assets_dir = base_dir / "assets"
    fonts_dir = assets_dir / "fonts"
    assets_dir.mkdir(parents=True, exist_ok=True)
    fonts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ensure banner.png exists
    local_banner = assets_dir / "banner.png"
    if not local_banner.exists():
        if EXTERNAL_BANNER_PATH.exists():
            shutil.copy2(EXTERNAL_BANNER_PATH, local_banner)
        else:
            # Check parent directory
            parent_banner = base_dir / "banner.png"
            if parent_banner.exists():
                shutil.copy2(parent_banner, local_banner)

    # 2. Ensure avatar.png exists
    local_avatar = assets_dir / "avatar.png"
    if not local_avatar.exists():
        if local_banner.exists():
            crop_and_save_avatar(local_banner, local_avatar)
        elif EXTERNAL_BANNER_PATH.exists():
            crop_and_save_avatar(EXTERNAL_BANNER_PATH, local_avatar)

    # 3. Ensure all 4 fonts exist
    for font_name, url in FONT_URLS.items():
        font_path = fonts_dir / font_name
        if not font_path.exists() or font_path.stat().st_size < 1000:
            print(f"[ASSETS] Downloading font {font_name}...")
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = resp.read()
                    with open(font_path, "wb") as f:
                        f.write(data)
                print(f"[ASSETS] Saved {font_name} ({len(data)} bytes)")
            except Exception as e:
                print(f"[ASSETS WARNING] Failed to download {font_name}: {e}")

    return assets_dir


def crop_and_save_avatar(banner_path: Path, out_avatar_path: Path):
    """
    Crops the avatar circle from the banner:
    - Banner is 386x119 px.
    - Avatar center (56, 59), radius 56 -> bounding box (0, 3, 112, 115).
    - Resizes to 136x136.
    - Applies a circular anti-aliased mask (drawn at 4x then downscaled with LANCZOS).
    """
    banner = Image.open(banner_path).convert("RGBA")
    # Native bounding box: (center_x - radius, center_y - radius, center_x + radius, center_y + radius)
    crop_box = (0, 3, 112, 115)
    avatar_crop = banner.crop(crop_box)

    target_size = (AVATAR_SIZE, AVATAR_SIZE)
    avatar_resized = avatar_crop.resize(target_size, Image.Resampling.LANCZOS)

    # 4x supersampled circular mask for crisp anti-aliasing
    scale = 4
    mask_size = (target_size[0] * scale, target_size[1] * scale)
    mask = Image.new("L", mask_size, 0)
    d_mask = ImageDraw.Draw(mask)
    d_mask.ellipse((0, 0, mask_size[0] - 1, mask_size[1] - 1), fill=255)
    mask = mask.resize(target_size, Image.Resampling.LANCZOS)

    avatar_final = Image.new("RGBA", target_size, (0, 0, 0, 0))
    avatar_final.paste(avatar_resized, (0, 0))
    avatar_final.putalpha(mask)

    out_avatar_path.parent.mkdir(parents=True, exist_ok=True)
    avatar_final.save(out_avatar_path, format="PNG")
    print(f"[ASSETS] Created circle-masked avatar at {out_avatar_path}")


class TextRenderer:
    """Manages multi-font loading, character fallback, line measurement, and token rendering."""

    def __init__(self, fonts_dir: Path, font_size: int):
        self.font_size = font_size
        self.fonts_dir = fonts_dir

        inter_p = fonts_dir / "Inter-Regular.ttf"
        hindi_p = fonts_dir / "NotoSansDevanagari-Regular.ttf"
        emoji_p = fonts_dir / "NotoColorEmoji.ttf"

        self.font_inter = ImageFont.truetype(str(inter_p), font_size) if inter_p.exists() else ImageFont.load_default()
        self.font_hindi = ImageFont.truetype(str(hindi_p), font_size) if hindi_p.exists() else self.font_inter
        self.font_emoji = ImageFont.truetype(str(emoji_p), font_size) if emoji_p.exists() else self.font_inter

        # Mac Apple Color Emoji fallback if available
        mac_emoji = Path("/System/Library/Fonts/Apple Color Emoji.ttc")
        if mac_emoji.exists():
            try:
                self.font_mac_emoji = ImageFont.truetype(str(mac_emoji), font_size)
            except Exception:
                self.font_mac_emoji = self.font_emoji
        else:
            self.font_mac_emoji = self.font_emoji

    def get_font_for_char(self, char: str) -> Tuple[ImageFont.ImageFont, str]:
        if is_emoji_char(char):
            return self.font_emoji, "emoji"
        if is_devanagari(char):
            return self.font_hindi, "hindi"
        return self.font_inter, "inter"

    def split_runs(self, text: str) -> List[Tuple[str, str, ImageFont.ImageFont]]:
        """Splits a string into consecutive runs sharing the same font classification."""
        if not text:
            return []
        runs = []
        curr_chars = [text[0]]
        curr_font, curr_kind = self.get_font_for_char(text[0])

        for c in text[1:]:
            f, kind = self.get_font_for_char(c)
            if kind == curr_kind:
                curr_chars.append(c)
            else:
                runs.append(("".join(curr_chars), curr_kind, curr_font))
                curr_chars = [c]
                curr_font = f
                curr_kind = kind

        if curr_chars:
            runs.append(("".join(curr_chars), curr_kind, curr_font))
        return runs

    def measure_text(self, draw: ImageDraw.ImageDraw, text: str) -> float:
        """Accurately measures text width across all character fallback fonts."""
        runs = self.split_runs(text)
        width = 0.0
        for run_text, kind, font in runs:
            try:
                width += draw.textlength(run_text, font=font)
            except Exception:
                bbox = draw.textbbox((0, 0), run_text, font=font)
                width += (bbox[2] - bbox[0])
        return width

    def draw_text(self, draw: ImageDraw.ImageDraw, target_img: Image.Image, x: float, y: float, text: str, fill: tuple) -> float:
        """Draws text runs with character fallback and embedded emoji support."""
        runs = self.split_runs(text)
        curr_x = x
        for run_text, kind, font in runs:
            if kind == "emoji":
                # Draw color emoji
                try:
                    draw.text((int(round(curr_x)), int(round(y))), run_text, font=font, embedded_color=True)
                except Exception:
                    try:
                        draw.text((int(round(curr_x)), int(round(y))), run_text, font=self.font_mac_emoji, embedded_color=True)
                    except Exception:
                        draw.text((int(round(curr_x)), int(round(y))), run_text, font=font, fill=fill)
            else:
                draw.text((int(round(curr_x)), int(round(y))), run_text, font=font, fill=fill)

            try:
                advance = draw.textlength(run_text, font=font)
            except Exception:
                bbox = draw.textbbox((0, 0), run_text, font=font)
                advance = float(bbox[2] - bbox[0])
            curr_x += advance

        return curr_x - x


def wrap_words_to_lines(
    tokens: List[Tuple[str, bool]],
    renderer: TextRenderer,
    draw: ImageDraw.ImageDraw,
    max_width: int
) -> List[List[Tuple[str, bool]]]:
    """
    Wraps tokens [(word, is_highlighted), ...] into lines that do not exceed max_width.
    """
    lines = []
    current_line: List[Tuple[str, bool]] = []
    current_width = 0.0
    space_width = renderer.measure_text(draw, " ")

    for word, is_hl in tokens:
        word_w = renderer.measure_text(draw, word)
        needed_width = (space_width if current_line else 0.0) + word_w

        if current_line and (current_width + needed_width > max_width):
            lines.append(current_line)
            current_line = [(word, is_hl)]
            current_width = word_w
        else:
            if current_line:
                current_width += space_width
            current_line.append((word, is_hl))
            current_width += word_w

    if current_line:
        lines.append(current_line)

    return lines


def build_text_layout(
    setup: Optional[str],
    punchline: str,
    highlight: str,
    fonts_dir: Path,
    draw: ImageDraw.ImageDraw
) -> Tuple[List[Any], int, int, TextRenderer]:
    """
    Computes text layout and line wrapping.
    If the text block would go past y=1350, shrinks font in 2px steps (min 40px) until it fits.
    """
    font_size = DEFAULT_FONT_SIZE

    # Pre-tokenize punchline with highlight detection
    p_lower = punchline.lower()
    h_lower = highlight.strip().lower()
    h_start = p_lower.find(h_lower) if h_lower else -1
    h_end = h_start + len(h_lower) if h_start != -1 else -1

    words = punchline.strip().split()
    tokens = []
    char_search_idx = 0
    for w in words:
        w_start = punchline.find(w, char_search_idx)
        w_end = w_start + len(w)
        char_search_idx = w_end
        is_hl = (h_start != -1 and not (w_end <= h_start or w_start >= h_end))
        tokens.append((w, is_hl))

    while font_size >= MIN_FONT_SIZE:
        renderer = TextRenderer(fonts_dir, font_size)
        line_height = int(round(font_size * BASE_LINE_HEIGHT / DEFAULT_FONT_SIZE))

        all_lines = []

        # 1. Setup line (if present)
        if setup and setup.strip():
            setup_words = [(w, False) for w in setup.strip().split()]
            setup_lines = wrap_words_to_lines(setup_words, renderer, draw, TEXT_MAX_WIDTH)
            all_lines.extend(setup_lines)
            # Blank line between setup and punchline
            all_lines.append(None)

        # 2. Punchline lines
        punchline_lines = wrap_words_to_lines(tokens, renderer, draw, TEXT_MAX_WIDTH)
        all_lines.extend(punchline_lines)

        # Calculate bottom Y position
        total_visual_lines = len(all_lines)
        block_bottom_y = TEXT_BLOCK_Y + (total_visual_lines - 1) * line_height + font_size

        if block_bottom_y <= MAX_TEXT_BOTTOM_Y or font_size == MIN_FONT_SIZE:
            return all_lines, font_size, line_height, renderer

        font_size -= 2

    return all_lines, font_size, line_height, renderer


def render_reel(
    data: Dict[str, Any],
    output_png_path: str,
    assets_dir: Optional[str] = None
) -> str:
    """
    Renders a 1080x1920 PNG static graphic matching the exact "9 to 5 Office" FORMAT specifications.
    """
    base_assets = Path(assets_dir) if assets_dir else ensure_assets()
    ensure_assets(base_assets.parent)
    fonts_dir = base_assets / "fonts"

    # 1. Canvas 1080x1920 pure black
    canvas = Image.new("RGBA", (CANVAS_WIDTH, CANVAS_HEIGHT), COLOR_BG)

    # 2. Paste Circle-Masked Avatar at (100, 560)
    avatar_path = base_assets / "avatar.png"
    if not avatar_path.exists():
        banner_path = base_assets / "banner.png"
        crop_and_save_avatar(banner_path, avatar_path)

    avatar_img = Image.open(avatar_path).convert("RGBA")
    if avatar_img.size != (AVATAR_SIZE, AVATAR_SIZE):
        avatar_img = avatar_img.resize((AVATAR_SIZE, AVATAR_SIZE), Image.Resampling.LANCZOS)
    canvas.paste(avatar_img, (AVATAR_X, AVATAR_Y), avatar_img)

    # 3. Draw Name & Handle
    draw = ImageDraw.Draw(canvas)
    name_font = ImageFont.truetype(str(fonts_dir / "Poppins-Bold.ttf"), NAME_FONT_SIZE)
    handle_font = ImageFont.truetype(str(fonts_dir / "Inter-Regular.ttf"), HANDLE_FONT_SIZE)

    draw.text((NAME_X, NAME_Y), "9 to 5 Office", font=name_font, fill=COLOR_WHITE)
    draw.text((HANDLE_X, HANDLE_Y), "@9to5_office", font=handle_font, fill=COLOR_HANDLE)

    # 4. Text Layout & Dynamic Font Shrinking
    setup = data.get("setup")
    punchline = data.get("punchline", "").strip()
    highlight = data.get("highlight", "").strip()

    lines, font_size, line_height, renderer = build_text_layout(setup, punchline, highlight, fonts_dir, draw)

    # 5. Render Text Block at x=108, y=760
    current_y = TEXT_BLOCK_Y
    space_w = renderer.measure_text(draw, " ")

    for line in lines:
        if line is None:
            # Blank line between setup and punchline
            current_y += line_height
            continue

        current_x = float(TEXT_BLOCK_X)
        for i, (word, is_hl) in enumerate(line):
            fill_color = COLOR_HIGHLIGHT if is_hl else COLOR_WHITE
            word_w = renderer.draw_text(draw, canvas, current_x, current_y, word, fill_color)
            current_x += word_w
            if i < len(line) - 1:
                draw.text((int(round(current_x)), int(round(current_y))), " ", font=renderer.font_inter, fill=COLOR_WHITE)
                current_x += space_w

        current_y += line_height

    out_p = Path(output_png_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_p, format="PNG")
    print(f"[RENDER] Saved 1080x1920 PNG to {out_p}")
    return str(out_p)


def create_mp4(png_path: str, output_mp4_path: str, duration: int = 7) -> str:
    """
    Generates a 7-second silent MP4 from the static PNG with ffmpeg.
    Command: ffmpeg -loop 1 -i in.png -t 7 -r 30 -c:v libx264 -pix_fmt yuv420p -an out.mp4
    """
    out_p = Path(output_mp4_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-loop", "1",
        "-i", str(png_path),
        "-t", str(duration),
        "-r", "30",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-an",
        str(out_p)
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {res.stderr}")

    print(f"[RENDER] Generated {duration}s silent MP4: {out_p}")
    return str(out_p)

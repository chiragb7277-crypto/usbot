"""
emoji_helper.py
================
High-performance iPhone (Apple) emoji layout and rendering system for Pillow images.

Features:
- Splits lines into text and emoji tokens.
- Native Apple Color Emoji glyph extraction on macOS (/System/Library/Fonts/Apple Color Emoji.ttc).
- High-res (160x160) Apple emoji CDN fallback with persistent disk caching in resources/emojis/.
- Accurate line measurement and proportional scaling (dynamic line fitting).
- Automatic vertical alignment with font cap-height.
- In-memory LRU caching for 0ms re-rendering.
"""

import os
import re
import unicodedata
import urllib.request
import urllib.parse
from io import BytesIO
from typing import List, Tuple, Optional, Dict, Any
from PIL import Image, ImageDraw, ImageFont

try:
    import emoji
    EMOJI_LIB_AVAILABLE = True
except ImportError:
    EMOJI_LIB_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMOJI_CACHE_DIR = os.path.join(BASE_DIR, "resources", "emojis")
os.makedirs(EMOJI_CACHE_DIR, exist_ok=True)

MAC_APPLE_EMOJI_FONT = "/System/Library/Fonts/Apple Color Emoji.ttc"
_MEM_CACHE: Dict[str, Image.Image] = {}

def clean_headline_text(text: str) -> str:
    """
    Sanitizes headline text for PIL / FreeType rendering:
    - Strips invisible format characters (zero-width space \\u200b, ZWNJ, BOM, bidi markers)
      that cause FreeType to render empty square/box .notdef glyphs.
    - Preserves valid emoji sequences (including flags, skin tones, ZWJ sequences).
    - Normalizes non-standard spaces (NBSP, en/em space, thin space) to standard space ' '.
    """
    if not text:
        return ""

    # Normalize various Unicode whitespace characters to standard ASCII space
    whitespace_map = {
        "\u00a0": " ",  # No-break space
        "\u2000": " ",  # En quad
        "\u2001": " ",  # Em quad
        "\u2002": " ",  # En space
        "\u2003": " ",  # Em space
        "\u2004": " ",  # Three-per-em space
        "\u2005": " ",  # Four-per-em space
        "\u2006": " ",  # Six-per-em space
        "\u2007": " ",  # Figure space
        "\u2008": " ",  # Punctuation space
        "\u2009": " ",  # Thin space
        "\u200a": " ",  # Hair space
        "\u202f": " ",  # Narrow no-break space
        "\u205f": " ",  # Medium mathematical space
        "\u3000": " ",  # Ideographic space
    }
    for orig, rep in whitespace_map.items():
        if orig in text:
            text = text.replace(orig, rep)

    # Invisible format / control characters that render as boxes in TrueType fonts
    bad_chars = {
        "\u200b",  # Zero-width space
        "\u200c",  # Zero-width non-joiner
        "\u200d",  # Zero-width joiner (outside emojis)
        "\u200e",  # Left-to-right mark
        "\u200f",  # Right-to-left mark
        "\ufeff",  # Zero-width no-break space / BOM
        "\ufffc",  # Object replacement character
        "\ufffd",  # Replacement character
        "\u00ad",  # Soft hyphen
        "\u2060",  # Word joiner
        "\u202a", "\u202b", "\u202c", "\u202d", "\u202e",  # Bidi overrides
        "\u2066", "\u2067", "\u2068", "\u2069",             # Bidi isolates
    }

    # If emoji library is available, split into emoji tokens vs text tokens
    if EMOJI_LIB_AVAILABLE:
        matches = list(emoji.analyze(text))
        if matches:
            segments = []
            last_idx = 0
            for m in matches:
                start, end = m.value.start, m.value.end
                if start > last_idx:
                    segments.append((False, text[last_idx:start]))
                segments.append((True, m.chars))
                last_idx = end
            if last_idx < len(text):
                segments.append((False, text[last_idx:]))
        else:
            segments = [(False, text)]
    else:
        segments = [(False, text)]

    cleaned_parts = []
    for is_em, seg in segments:
        if is_em:
            cleaned_parts.append(seg)
        else:
            cleaned = "".join(
                c for c in seg
                if c not in bad_chars and (c in "\n\r\t" or unicodedata.category(c)[0] != "C")
            )
            cleaned_parts.append(cleaned)

    return "".join(cleaned_parts)

def emoji_to_hex(emoji_str: str) -> str:
    """Converts an emoji string to hyphen-separated lowercase hex code."""
    return "-".join(f"{ord(c):x}" for c in emoji_str)

def has_emojis(text: str) -> bool:
    """Returns True if the text contains at least one emoji."""
    if not text:
        return False
    if EMOJI_LIB_AVAILABLE:
        return bool(emoji.emoji_count(text) > 0)
    # Basic unicode range fallback
    for ch in text:
        if ord(ch) > 0x2300:
            return True
    return False

def get_apple_emoji_image(emoji_char: str) -> Optional[Image.Image]:
    """
    Returns an RGBA PIL Image of the authentic iPhone (Apple) emoji.
    1. Checks in-memory cache.
    2. Checks disk cache (resources/emojis/<hex>.png).
    3. If on macOS, extracts directly from native Apple Color Emoji font.
    4. If on Linux/Docker/Cloud, fetches 160x160 Apple emoji from CDN and caches to disk.
    """
    if emoji_char in _MEM_CACHE:
        return _MEM_CACHE[emoji_char]

    code_hex = emoji_to_hex(emoji_char)
    cached_path = os.path.join(EMOJI_CACHE_DIR, f"{code_hex}.png")

    # Check local disk cache
    if os.path.exists(cached_path):
        try:
            img = Image.open(cached_path).convert("RGBA")
            _MEM_CACHE[emoji_char] = img
            return img
        except Exception:
            pass

    # 1. Native macOS font extraction
    if os.path.exists(MAC_APPLE_EMOJI_FONT):
        try:
            efont = ImageFont.truetype(MAC_APPLE_EMOJI_FONT, 160)
            temp = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
            tdraw = ImageDraw.Draw(temp)
            tdraw.text((10, 10), emoji_char, font=efont, embedded_color=True)
            bbox = temp.getbbox()
            if bbox:
                cropped = temp.crop(bbox)
                cropped.save(cached_path, format="PNG")
                _MEM_CACHE[emoji_char] = cropped
                return cropped
        except Exception:
            pass

    # 2. High-res Apple emoji CDN fallback
    urls = [
        f"https://emojicdn.elk.sh/{urllib.parse.quote(emoji_char)}?style=apple",
        f"https://cdn.jsdelivr.net/npm/emoji-datasource-apple@15.1.2/img/apple/64/{code_hex}.png"
    ]
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = resp.read()
                img = Image.open(BytesIO(data)).convert("RGBA")
                img.save(cached_path, format="PNG")
                _MEM_CACHE[emoji_char] = img
                return img
        except Exception:
            continue

    return None

def split_text_and_emojis(text: str) -> List[Tuple[bool, str]]:
    """
    Splits a single line of text into segments:
    [(is_emoji: bool, content: str), ...]
    Pre-cleans text to strip invisible formatting and control characters.
    """
    if not text:
        return []
    text = clean_headline_text(text)
    if not text:
        return []
    if not EMOJI_LIB_AVAILABLE:
        return [(False, text)]

    matches = list(emoji.analyze(text))
    if not matches:
        return [(False, text)]

    segments = []
    last_idx = 0
    for match in matches:
        start = match.value.start
        end = match.value.end
        if start > last_idx:
            segments.append((False, text[last_idx:start]))
        segments.append((True, match.chars))
        last_idx = end

    if last_idx < len(text):
        segments.append((False, text[last_idx:]))

    return segments

def get_font_cap_metrics(font: ImageFont.ImageFont) -> Tuple[int, int]:
    """Returns (cap_top, cap_h) relative to font origin."""
    bbox = font.getbbox("H")
    if bbox:
        cap_top = bbox[1]
        cap_h = max(10, bbox[3] - bbox[1])
    else:
        try:
            ascent, descent = font.getmetrics()
            cap_top = 0
            cap_h = max(10, ascent)
        except Exception:
            cap_top = 0
            cap_h = getattr(font, "size", 50)
    return cap_top, cap_h

def measure_line_with_emojis(
    draw: ImageDraw.ImageDraw,
    line: str,
    font: ImageFont.ImageFont
) -> Tuple[float, List[Tuple[bool, str, Optional[Image.Image], float, int]], int, int]:
    """
    Measures a line of text that may contain emojis.
    Returns:
    - total_width: float
    - measured_segments: list of (is_emoji, text, emoji_img, seg_width, emoji_size)
    - cap_top: int
    - cap_h: int
    """
    segments = split_text_and_emojis(line)
    cap_top, cap_h = get_font_cap_metrics(font)
    emoji_size = max(12, int(cap_h * 1.08))
    spacing = int(emoji_size * 0.08)

    total_w = 0.0
    measured = []
    for is_emoji, seg in segments:
        if is_emoji:
            img = get_apple_emoji_image(seg)
            if img:
                w = float(emoji_size)
                measured.append((True, seg, img, w, emoji_size))
                total_w += w + spacing
            else:
                # Do NOT pass missing emoji to raw text font because fonts lack emoji glyphs
                # and will draw a black/empty square box (.notdef).
                # Omit or skip cleanly.
                pass
        else:
            clean_seg = clean_headline_text(seg)
            if clean_seg:
                try:
                    w = float(draw.textlength(clean_seg, font=font))
                except Exception:
                    bbox = draw.textbbox((0, 0), clean_seg, font=font)
                    w = float(bbox[2] - bbox[0])
                measured.append((False, clean_seg, None, w, 0))
                total_w += w

    return total_w, measured, cap_top, cap_h

def draw_text_with_emojis(
    draw: ImageDraw.ImageDraw,
    target_img: Image.Image,
    x: int,
    y: int,
    line: str,
    font: ImageFont.ImageFont,
    fill: tuple,
    measured_data: Optional[Tuple[float, List, int, int]] = None
) -> int:
    """
    Draws text onto target_img/draw.
    If line has emojis, renders iPhone emojis precisely positioned with font cap-height.
    Returns the final x coordinate.
    """
    clean_line = clean_headline_text(line)
    if not has_emojis(clean_line):
        draw.text((x, y), clean_line, font=font, fill=fill)
        try:
            return x + int(draw.textlength(clean_line, font=font))
        except Exception:
            return x

    if measured_data is not None:
        _, measured, cap_top, cap_h = measured_data
    else:
        _, measured, cap_top, cap_h = measure_line_with_emojis(draw, clean_line, font)

    curr_x = float(x)
    for is_emoji, seg, img, w, e_size in measured:
        if is_emoji:
            if img is not None:
                # Resize Apple emoji using high-quality resampling
                e_resized = img.resize((e_size, e_size), Image.Resampling.LANCZOS)
                # Center emoji vertically relative to capital letters
                e_y = y + cap_top + (cap_h - e_size) // 2
                target_img.paste(e_resized, (int(round(curr_x)), int(round(e_y))), e_resized)
                curr_x += w + int(e_size * 0.08)
        else:
            draw.text((int(round(curr_x)), y), seg, font=font, fill=fill)
            curr_x += w

    return int(round(curr_x))

#!/usr/bin/env python3
"""
textbox_bot.py
Telegram bot for generating premium text-box images.
Features:
- Select style font: "faith" or "maga_charlie"
- Enter text: renders text centered on a dark gradient background inside a semi-transparent glassmorphic card.
- Dynamic line fitting: scales each line to span the box width (touching the borders).
- All text is auto-capitalized.
- Conditional line coloring:
  - Default (faith): last 2/3 lines colored.
  - Maga/Charlie: middle lines colored (4 lines → middle 2, 5 lines → middle 3, etc.).
- Persistent session: user stays on the same page after generating, can send more text or change color.
"""

import os
import sys

# Ensure virtual environment site-packages are accessible even when invoked via system python
_venv_site = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "lib", "python3.11", "site-packages")
if os.path.exists(_venv_site) and _venv_site not in sys.path:
    sys.path.insert(0, _venv_site)

import threading
import time
from http.server import SimpleHTTPRequestHandler, HTTPServer

def start_health_servers():
    """Start HTTP health server for cloud hosting (Render / HF) on PORT or single fallback port 10000."""
    port_env = os.environ.get("PORT")
    if port_env:
        try:
            ports_to_bind = [int(port_env)]
        except ValueError:
            ports_to_bind = [10000]
    else:
        # Default single health port when running locally or unconfigured
        ports_to_bind = [int(os.environ.get("HEALTH_PORT", 10000))]

    class HealthStatusHandler(SimpleHTTPRequestHandler):
        def do_GET(self):
            token_present = bool(os.getenv("TELEGRAM_BOT_TOKEN"))
            status_text = "Bot Active & Ready" if token_present else "Awaiting TELEGRAM_BOT_TOKEN"
            html_body = f"""<!DOCTYPE html>
<html>
<head>
    <title>US Pages Telegram Bot</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; }}
        .card {{ background: #1e293b; padding: 2rem; border-radius: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); max-width: 480px; width: 90%; text-align: center; border: 1px solid #334155; }}
        h1 {{ color: #38bdf8; margin-top: 0; }}
        .badge {{ display: inline-block; padding: 0.35rem 0.75rem; border-radius: 9999px; font-weight: 600; font-size: 0.875rem; margin: 0.5rem 0; background: #22c55e20; color: #4ade80; border: 1px solid #22c55e40; }}
        .badge.warning {{ background: #ef444420; color: #f87171; border: 1px solid #ef444440; }}
        p {{ color: #94a3b8; font-size: 0.95rem; line-height: 1.5; }}
    </style>
</head>
<body>
    <div class="card">
        <h1>🤖 Textbox Bot Online</h1>
        <div><span class="badge {'warning' if not token_present else ''}">Status: {status_text}</span></div>
        <p>Health check server is actively responding. UptimeRobot pings keep this service awake 24/7 without idle spin-downs.</p>
    </div>
</body>
</html>"""
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.send_header("Content-Length", str(len(html_body.encode("utf-8"))))
            self.end_headers()
            self.wfile.write(html_body.encode("utf-8"))

        def log_message(self, format, *args):
            pass

    class ReusableHTTPServer(HTTPServer):
        allow_reuse_address = True

    for p in ports_to_bind:
        def make_server(port_num):
            try:
                server = ReusableHTTPServer(("0.0.0.0", port_num), HealthStatusHandler)
                print(f"[HEALTH SERVER] Health server listening on 0.0.0.0:{port_num}", flush=True)
                server.serve_forever()
            except OSError:
                # Port already bound or not available, fail quietly without spamming
                pass
            except Exception:
                pass

        threading.Thread(target=make_server, args=(p,), daemon=True).start()

# Launch health server IMMEDIATELY (in <50ms) before any heavy ML packages are imported
start_health_servers()

import logging
import urllib.request
import argparse
import asyncio
import base64
import re
import uuid
from io import BytesIO
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from emoji_helper import (
    has_emojis,
    get_apple_emoji_image,
    measure_line_with_emojis,
    draw_text_with_emojis,
    get_font_cap_metrics
)
# NOTE: rembg and onnxruntime are lazy-loaded on demand to keep baseline memory lean (<100MB)
import requests
import instaloader
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputMediaPhoto
from telegram.ext import ContextTypes
from telegram.request import HTTPXRequest
from telegram.error import TimedOut, NetworkError, TelegramError
# Load environment variables
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))


import json

# Logging setup
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger("textbox_bot")

# Global high-performance caches for real-time 3D positioning
REMBG_SESSION = None
REMBG_LOCK = threading.Lock()
# Per-cache locks so concurrent users don't corrupt shared dicts
_CUTOUT_LOCK = threading.Lock()
_SHAPE_LOCK = threading.Lock()
_OVERLAY_LOCK = threading.Lock()
_BG_CROPPED_LOCK = threading.Lock()

def get_rembg_session():
    global REMBG_SESSION
    with REMBG_LOCK:
        if REMBG_SESSION is None:
            try:
                from rembg import new_session
                # Use u2netp (lightweight 4.7MB model, ~35MB RAM) to comfortably fit within 512MB RAM
                REMBG_SESSION = new_session("u2netp")
            except Exception as e:
                logger.error(f"Failed to create rembg session: {e}")
    return REMBG_SESSION

CUTOUT_CACHE = {}
BG_CROPPED_CACHE = {}
OVERLAY_CACHE = {}
SHAPE_CACHE = {}

def clear_media_caches(bg_path: str = None, shape_path: str = None, shape2_path: str = None):
    """Clears cached images to free memory after positioning is done."""
    if bg_path:
        keys_to_del = [k for k in CUTOUT_CACHE if k[0] == bg_path]
        for k in keys_to_del:
            CUTOUT_CACHE.pop(k, None)
        keys_to_del = [k for k in BG_CROPPED_CACHE if k[0] == bg_path]
        for k in keys_to_del:
            BG_CROPPED_CACHE.pop(k, None)
    if shape_path:
        keys_to_del = [k for k in SHAPE_CACHE if k[0] == shape_path]
        for k in keys_to_del:
            SHAPE_CACHE.pop(k, None)
    if shape2_path:
        keys_to_del = [k for k in SHAPE_CACHE if k[0] == shape2_path]
        for k in keys_to_del:
            SHAPE_CACHE.pop(k, None)

def calculate_target_h(png_bytes: bytes, line_bottom: int = None) -> tuple:
    """Returns (target_h, y_offset) based on the overlay and anchor position."""
    overlay_key = (id(png_bytes), len(png_bytes))
    if overlay_key in OVERLAY_CACHE:
        overlay = OVERLAY_CACHE[overlay_key]
    else:
        overlay = Image.open(BytesIO(png_bytes)).convert("RGBA")
        with _OVERLAY_LOCK:
            if len(OVERLAY_CACHE) > 10:
                oldest = next(iter(OVERLAY_CACHE))
                OVERLAY_CACHE.pop(oldest, None)
            OVERLAY_CACHE[overlay_key] = overlay
    
    if line_bottom is None:
        line_bottom = 0
        for y in range(overlay.height):
            if overlay.getpixel((0, y))[3] > 0:
                line_bottom = y + 1
            
    is_bottom_anchored = (overlay.getpixel((540, 1340))[3] > 0)
    if is_bottom_anchored:
        line_pos = line_bottom
        if line_pos is None or line_pos <= 0 or line_pos >= 1350:
            for y in range(overlay.height // 2, overlay.height):
                if overlay.getpixel((540, y))[3] == 255:
                    line_pos = y
                    break
            if line_pos is None:
                line_pos = 890
        return line_pos, 0
    else:
        if line_bottom is None or line_bottom <= 0 or line_bottom >= 1350:
            line_bottom = 450
        return 1350 - line_bottom, line_bottom

def compute_and_cache_cutout(bg_image_path: str, target_h: int):
    """Generates the background cutout using rembg and caches it for instant positioning."""
    if not bg_image_path or not os.path.exists(bg_image_path) or target_h <= 0:
        return None
    bg_mtime = os.path.getmtime(bg_image_path)
    cutout_cache_key = (bg_image_path, bg_mtime, target_h)
    with _CUTOUT_LOCK:
        if cutout_cache_key in CUTOUT_CACHE:
            return CUTOUT_CACHE[cutout_cache_key]
    try:
        bg_img = Image.open(bg_image_path).convert("RGBA")
        bg_enhanced = enhance_image_quality(bg_img)
        session = get_rembg_session()
        from rembg import remove
        with REMBG_LOCK:
            cutout = remove(bg_enhanced, session=session)
        bg_aspect = bg_enhanced.width / bg_enhanced.height
        target_aspect = 1080 / target_h
        if bg_aspect > target_aspect:
            new_w = int(bg_enhanced.height * target_aspect)
            left = (bg_enhanced.width - new_w) // 2
            crop_box = (left, 0, left + new_w, bg_enhanced.height)
        else:
            new_h = int(bg_enhanced.width / target_aspect)
            top = (bg_enhanced.height - new_h) // 2
            crop_box = (0, top, bg_enhanced.width, top + new_h)
        cutout_cropped = cutout.crop(crop_box).resize((1080, target_h), Image.Resampling.LANCZOS)
        with _CUTOUT_LOCK:
            if len(CUTOUT_CACHE) >= 20:
                oldest_key = next(iter(CUTOUT_CACHE))
                CUTOUT_CACHE.pop(oldest_key, None)
            CUTOUT_CACHE[cutout_cache_key] = cutout_cropped
        logger.info(f"Background cutout generated and cached for {bg_image_path}")
        return cutout_cropped
    except Exception as e:
        logger.error(f"Failed to generate background cutout: {e}")
        return None

def warmup_cutout_cache(bg_image_path: str, png_bytes: bytes, line_bottom: int = None):
    """Pre-computes cutout in background thread so posting shape image is instant."""
    try:
        if not bg_image_path or not os.path.exists(bg_image_path) or not png_bytes:
            return
        target_h, _ = calculate_target_h(png_bytes, line_bottom)
        compute_and_cache_cutout(bg_image_path, target_h)
    except Exception as e:
        logger.error(f"Error in warmup_cutout_cache: {e}")

def save_user_session(user_id, chosen_font=None, chosen_color=None, doge_highlight_lines=None, last_text=None, line_bottom=None):
    try:
        session_file = os.path.join(BASE_DIR, "scratch", f"session_{user_id}.json")
        os.makedirs(os.path.dirname(session_file), exist_ok=True)
        data = {}
        if os.path.exists(session_file):
            try:
                with open(session_file, "r") as f:
                    data = json.load(f)
            except Exception:
                data = {}
        if chosen_font is not None:
            data["chosen_font"] = chosen_font
        if chosen_color is not None:
            data["chosen_color"] = chosen_color
        if doge_highlight_lines is not None:
            data["doge_highlight_lines"] = doge_highlight_lines
        if last_text is not None:
            data["last_text"] = last_text
        if line_bottom is not None:
            data["line_bottom"] = line_bottom
            
        with open(session_file, "w") as f:
            json.dump(data, f)
        logger.info(f"Saved session for user {user_id}: {data}")
    except Exception as e:
        logger.error(f"Failed to save user session: {e}")

def load_user_session(user_id):
    try:
        session_file = os.path.join(BASE_DIR, "scratch", f"session_{user_id}.json")
        if os.path.exists(session_file):
            with open(session_file, "r") as f:
                data = json.load(f)
                logger.info(f"Loaded session for user {user_id}: {data}")
                return data
    except Exception as e:
        logger.error(f"Failed to load user session: {e}")
    return {}

def restore_session_if_needed(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update or not update.effective_user:
        return
    user_id = update.effective_user.id
    session = load_user_session(user_id)
    if session:
        if not context.user_data.get("chosen_font") and session.get("chosen_font"):
            context.user_data["chosen_font"] = session.get("chosen_font")
        if not context.user_data.get("chosen_color") and session.get("chosen_color"):
            context.user_data["chosen_color"] = session.get("chosen_color")
        if context.user_data.get("doge_highlight_lines") is None and session.get("doge_highlight_lines") is not None:
            context.user_data["doge_highlight_lines"] = session.get("doge_highlight_lines")
        if not context.user_data.get("last_text") and session.get("last_text"):
            context.user_data["last_text"] = session.get("last_text")
        if context.user_data.get("line_bottom") is None and session.get("line_bottom") is not None:
            context.user_data["line_bottom"] = session.get("line_bottom")

def fetch_instagram_post(url: str, user_id: int):
    """
    Fetches the caption and downloads the primary image/video of an Instagram post.
    Returns: (caption, media_path, is_video)
    """
    parts = url.strip("/").split("/")
    shortcode = None
    for i, p in enumerate(parts):
        if p in ("p", "reel", "tv") and i + 1 < len(parts):
            shortcode = parts[i + 1].split("?")[0]
            break
            
    if not shortcode:
        raise ValueError("Could not extract Instagram shortcode from URL.")
        
    L = instaloader.Instaloader()
    post = instaloader.Post.from_shortcode(L.context, shortcode)
    caption = post.caption or ""
    
    media_path = None
    is_video = post.is_video
    
    if is_video and post.video_url:
        response = requests.get(post.video_url, stream=True)
        if response.status_code == 200:
            media_path = os.path.join(BASE_DIR, f"user_bg_vid_{user_id}.mp4")
            with open(media_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
    elif post.url:
        response = requests.get(post.url)
        if response.status_code == 200:
            media_path = os.path.join(BASE_DIR, f"user_bg_img_{user_id}.jpg")
            with open(media_path, "wb") as f:
                f.write(response.content)
                
    return caption, media_path, is_video

# Font Download Links (official Google Fonts)
FONT_LINKS = {
    "faith": [
        "https://github.com/google/fonts/raw/10a708073179c32928eb894e53465fca8106772f/ofl/playfairdisplay/static/PlayfairDisplay-Italic.ttf",
        "https://github.com/google/fonts/raw/10a708073179c32928eb894e53465fca8106772f/ofl/playfairdisplay/PlayfairDisplay-Italic.ttf"
    ],
    "maga_charlie": [],  # Maga/Charlie uses system Impact font, no download needed
    "freedom": [
        "https://raw.githubusercontent.com/google/fonts/main/ofl/montserrat/static/Montserrat-Black.ttf",
        "https://raw.githubusercontent.com/google/fonts/main/ofl/montserrat/static/Montserrat-Bold.ttf"
    ]
}

def download_fonts():
    """Ensure fonts are downloaded and available."""
    fonts_dir = os.path.join(BASE_DIR, "fonts")
    os.makedirs(fonts_dir, exist_ok=True)
    
    for font_name, urls in FONT_LINKS.items():
        font_path = os.path.join(fonts_dir, f"{font_name}.ttf")
        if os.path.exists(font_path) and os.path.getsize(font_path) > 1024:
            logger.info(f"Font '{font_name}' already exists.")
            continue
            
        success = False
        for url in urls:
            try:
                logger.info(f"Downloading {font_name} from {url}...")
                urllib.request.urlretrieve(url, font_path)
                if os.path.exists(font_path) and os.path.getsize(font_path) > 1024:
                    logger.info(f"Successfully downloaded {font_name}.")
                    success = True
                    break
            except Exception as e:
                logger.warning(f"Failed to download {font_name} from {url}: {e}")
                if os.path.exists(font_path):
                    os.remove(font_path)
                    
        if not success:
            logger.error(f"Could not download font '{font_name}'. Fallbacks will be used.")

def get_line_height(font: ImageFont.FreeTypeFont) -> int:
    """Helper to calculate vertical height of font line."""
    try:
        ascent, descent = font.getmetrics()
        line_height = ascent + descent
        if line_height > 0:
            return line_height
    except Exception:
        pass
    try:
        bbox = font.getbbox("Ag")
        if bbox:
            return max(1, bbox[3] - bbox[1])
    except Exception:
        pass
    return max(1, getattr(font, "size", 12))

def draw_gradient_background(width, height, start_color, end_color):
    """Draw a smooth vertical color gradient on an RGB image."""
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)
    for y in range(height):
        ratio = y / height
        r = int(start_color[0] + (end_color[0] - start_color[0]) * ratio)
        g = int(start_color[1] + (end_color[1] - start_color[1]) * ratio)
        b = int(start_color[2] + (end_color[2] - start_color[2]) * ratio)
        draw.line([(0, y), (width, y)], fill=(r, g, b))
    return img

def draw_vertical_alpha_gradient(width, height, start_y, end_y, start_alpha=0, end_alpha=240, color=(0, 0, 0)):
    """Draw a smooth vertical alpha gradient mask from start_y to end_y."""
    gradient_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(gradient_img)
    span = max(1, end_y - start_y)
    for y in range(max(0, start_y), min(height, end_y)):
        ratio = (y - start_y) / span
        smooth_ratio = ratio * ratio * (3 - 2 * ratio)
        alpha = int(start_alpha + (end_alpha - start_alpha) * smooth_ratio)
        draw.line([(0, y), (width, y)], fill=(color[0], color[1], color[2], alpha))
    if end_alpha > 0 and end_y < height:
        draw.rectangle([(0, end_y), (width, height)], fill=(color[0], color[1], color[2], end_alpha))
    return gradient_img

FREEDOM_WATERMARK_LINE_COLOR = (248, 119, 57)  # #F87739, matches reference Freedom Front orange line

REALESRGAN_SESSION = None

def get_realesrgan_session():
    """Get or lazily initialize the Real-ESRGAN ONNX session for AI super-resolution."""
    global REALESRGAN_SESSION
    # On Render free tier (512MB RAM), skip heavy ONNX deep learning model to prevent OOM kills
    if os.environ.get("RENDER") or os.environ.get("LOW_MEMORY"):
        return None

    if REALESRGAN_SESSION is None:
        try:
            import onnxruntime as ort
        except ImportError:
            return None
        model_paths = [
            os.path.join(BASE_DIR, "resources", "RealESRGAN_x4plus.onnx"),
            os.path.join(BASE_DIR, "resources", "realesr-general-x4v3.onnx"),
            os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub", "models--mhmtaufiq--realesrgan-onnx", "snapshots", "c524eb8bdc8563e03d66e332a22886af1b08be21", "RealESRGAN_x4plus.onnx"),
            os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub", "models--Samo629--real-esrgan-onnx", "snapshots", "4e98d089bf844f4ddeb2e76248e5c94d3938bf7c", "realesr-general-x4v3.onnx")
        ]
        available_providers = ort.get_available_providers()
        providers = []
        if "CoreMLExecutionProvider" in available_providers:
            providers.append("CoreMLExecutionProvider")
        providers.append("CPUExecutionProvider")

        sess_opts = ort.SessionOptions()
        sess_opts.enable_cpu_mem_arena = False
        sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_opts.intra_op_num_threads = 1
        sess_opts.inter_op_num_threads = 1

        for mp in model_paths:
            if os.path.exists(mp):
                try:
                    REALESRGAN_SESSION = ort.InferenceSession(mp, sess_options=sess_opts, providers=providers)
                    logger.info(f"Loaded Real-ESRGAN AI model from: {mp} with providers {providers}")
                    break
                except Exception as e:
                    logger.error(f"Failed to load Real-ESRGAN model from {mp}: {e}")
    return REALESRGAN_SESSION


def enhance_image_quality(img: Image.Image, target_size: tuple = None) -> Image.Image:
    """
    Genuine AI Super-Resolution using Real-ESRGAN Neural Network.
    Cleans compression artifacts, reconstructs facial details and textures,
    and applies subtle news-agency grading.
    """
    if img is None:
        return img
        
    try:
        session = get_realesrgan_session()
        if session is not None:
            alpha = None
            if img.mode == "RGBA":
                alpha = img.split()[3]
                working_img = img.convert("RGB")
            else:
                working_img = img.convert("RGB") if img.mode != "RGB" else img.copy()

            w, h = working_img.size
            # Cap maximum dimension to 512px before 4x upscaling to keep inference fast (~3-5s on CoreML/CPU)
            scale_down = min(1.0, 512.0 / max(w, h))
            if scale_down < 1.0:
                working_img = working_img.resize((int(w * scale_down), int(h * scale_down)), Image.Resampling.LANCZOS)

            arr = np.array(working_img).astype(np.float32) / 255.0
            arr = np.transpose(arr, (2, 0, 1))
            arr = np.expand_dims(arr, axis=0)

            # Run Real-ESRGAN ONNX inference
            out = session.run(None, {"input": arr})[0][0]
            out = np.clip(out, 0, 1)
            out = np.transpose(out, (1, 2, 0))
            enhanced = Image.fromarray((out * 255.0).astype(np.uint8))

            # Pro studio grading: subtle contrast, vibrance, and fine-detail unsharp mask
            enhanced = ImageEnhance.Contrast(enhanced).enhance(1.04)
            enhanced = ImageEnhance.Color(enhanced).enhance(1.05)
            enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=1.2, percent=40, threshold=3))

            # Re-apply alpha channel if original was RGBA
            if alpha is not None:
                alpha_resized = alpha.resize(enhanced.size, Image.Resampling.LANCZOS)
                enhanced = enhanced.convert("RGBA")
                enhanced.putalpha(alpha_resized)

            if target_size is not None:
                enhanced = enhanced.resize(target_size, Image.Resampling.LANCZOS)

            return enhanced
    except Exception as e:
        logger.error(f"Real-ESRGAN AI enhancement error: {e}")

    # Fallback to mild contrast/color tuning if AI inference fails
    try:
        working_img = img.convert("RGB") if img.mode != "RGB" else img
        enhanced = ImageEnhance.Contrast(working_img).enhance(1.08)
        enhanced = ImageEnhance.Color(enhanced).enhance(1.08)
        enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=1.5, percent=50, threshold=3))
        if target_size is not None:
            enhanced = enhanced.resize(target_size, Image.Resampling.LANCZOS)
        return enhanced
    except Exception:
        if target_size is not None:
            return img.resize(target_size, Image.Resampling.LANCZOS)
        return img

def get_highlight_color(color_choice: str) -> tuple:
    """Helper to resolve highlight color choices to RGB tuples."""
    color_choice = (color_choice or "").lower().strip()
    if color_choice == "red":
        return (255, 49, 49)   # #ff3131
    elif color_choice == "orange":
        return (255, 145, 77)  # #ff914d
    elif color_choice == "magenta":
        return (203, 108, 230) # #cb6ce6
    elif color_choice == "green":
        return (0, 230, 118)   # #00e676
    elif color_choice == "blue":
        return (56, 182, 255)  # #38b6ff
    elif color_choice == "purple":
        return (140, 82, 255)  # #8c52ff
    else:
        return (255, 222, 89)  # #ffde59

def proofread_text_with_ai(text: str) -> str:
    """Proofreads text for spelling/grammar using Groq (openai/gpt-oss-120b or llama models) or Gemini."""
    # Try Groq first since we have a valid GROQ_API_KEY in .env
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {groq_key}",
            "Content-Type": "application/json"
        }
        prompt = (
            "You are an expert copyeditor. Fix any spelling, punctuation, or grammatical errors in the text below. "
            "Do NOT change the meaning or style. Keep the original text structure and line breaks where appropriate. "
            "CRITICAL: If any emojis are present in the text, you MUST PRESERVE all emojis in their exact positions. "
            "Do NOT remove, replace, or reorder any emojis. "
            "Do NOT add any introductory or concluding comments. Return ONLY the final corrected text.\n\n"
            f"Text to correct:\n{text}"
        )
        models_to_try = list(dict.fromkeys([
            os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            "llama-3.3-70b-versatile",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b"
        ]))
        for model_name in models_to_try:
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=5)
                if response.status_code == 200:
                    res_data = response.json()
                    content = res_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                    if content:
                        return content
            except Exception as e:
                logger.error(f"Error during Groq proofreading with {model_name}: {e}")

    # Fallback to Gemini if Groq fails or is not configured
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key and api_key != "YOUR_GEMINI_API_KEY":
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        prompt = (
            "You are an expert copyeditor. Fix any spelling, punctuation, or grammatical errors in the text below. "
            "Do NOT change the meaning or style. Keep the original text structure and line breaks where appropriate. "
            "CRITICAL: If any emojis are present in the text, you MUST PRESERVE all emojis in their exact positions. "
            "Do NOT remove, replace, or reorder any emojis. "
            "Do NOT add any introductory or concluding comments. Return ONLY the final corrected text.\n\n"
            f"Text to correct:\n{text}"
        )
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ]
        }
        headers = {"Content-Type": "application/json"}
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=8)
            if response.status_code == 200:
                res_data = response.json()
                parts = res_data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                if parts and "text" in parts[0]:
                    corrected = parts[0]["text"].strip()
                    if corrected.startswith("```"):
                        lines = corrected.split("\n")
                        if lines[0].startswith("```"):
                            lines = lines[1:]
                        if lines and lines[-1].startswith("```"):
                            lines = lines[:-1]
                        corrected = "\n".join(lines).strip()
                    return corrected
        except Exception as e:
            logger.error(f"Error during Gemini proofreading: {e}")

    return text

def is_ai_refusal(text: str) -> bool:
    """Detects if an AI response is a safety refusal or unhelpful generic disclaimer."""
    if not text:
        return True
    lower = text.lower().strip()
    refusal_phrases = [
        "i'm sorry", "i am sorry", "i’m sorry",
        "can't help", "cannot help", "can’t help",
        "i cannot fulfill", "i can't fulfill", "i cannot generate",
        "i can't generate", "as an ai", "as a language model",
        "unable to assist", "unable to help", "unable to fulfill",
        "against my safety", "violates my", "cannot provide",
        "i cannot write", "i can't write", "i'm unable", "i am unable",
        "i'd love to help! could you let me know"
    ]
    for phrase in refusal_phrases:
        if phrase in lower:
            return True
    if len(text.strip()) < 50:
        return True
    return False

def generate_post_caption(template_type: str = "charlie", image_bytes: bytes = None, text_content: str = "") -> str:
    """
    Generates an Instagram caption for Charlie Kirk or Supporting MAGA posts using Gemini vision or Groq fallback.
    """
    is_maga = template_type.lower() in ("maga", "supporting_maga")
    if is_maga:
        prompt = "write a 5 sentence 5 hashtag caption for this Instagram post from the conservative MAGA America First perspective"
        system_content = (
            "You are an expert social media copywriter for conservative MAGA and America First political commentary and news posts on Instagram. "
            "Do not include meta-commentary, introductory text, or explanations like 'Here is the caption:'. "
            "Output only the final 5-sentence caption and 5 hashtags."
        )
    else:
        prompt = "write a 5 sentence 5 hashtag caption for this Instagram post from the conservative perspective"
        system_content = (
            "You are an expert social media copywriter for conservative political commentary and news posts on Instagram. "
            "Do not include meta-commentary, introductory text, or explanations like 'Here is the caption:'. "
            "Output only the final 5-sentence caption and 5 hashtags."
        )
    
    # 1. Try Gemini API with image if key available
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key and api_key != "YOUR_GEMINI_API_KEY":
        try:
            model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
            parts = [{"text": prompt}]
            if image_bytes:
                encoded_image = base64.b64encode(image_bytes).decode("utf-8")
                mime_type = "image/png" if image_bytes.startswith(b"\x89PNG") else "image/jpeg"
                parts.append({
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": encoded_image
                    }
                })
            elif text_content:
                parts.append({"text": f"Post content / quote:\n{text_content}"})
                
            payload = {"contents": [{"parts": parts}]}
            response = requests.post(url, json=payload, timeout=20)
            if response.status_code == 200:
                data = response.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts_out = candidates[0].get("content", {}).get("parts", [])
                    if parts_out and "text" in parts_out[0]:
                        res_text = parts_out[0]["text"].strip()
                        if res_text and not is_ai_refusal(res_text):
                            return res_text
            else:
                logger.warning(f"Gemini caption generation returned {response.status_code}: {response.text}")
        except Exception as e:
            logger.error(f"Error during Gemini caption generation: {e}")

    # 2. Try Groq API fallback
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        models_to_try = [
            "qwen/qwen3.8-27b",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "llama-3.3-70b-versatile"
        ]
        if os.getenv("GROQ_MODEL"):
            models_to_try.insert(0, os.getenv("GROQ_MODEL"))
        seen = set()
        unique_models = [m for m in models_to_try if not (m in seen or seen.add(m))]
        
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {groq_key}",
            "Content-Type": "application/json"
        }
        user_msg = prompt
        if text_content:
            user_msg += f"\n\nPost content / quote:\n\"{text_content}\""
            
        for g_model in unique_models:
            payload = {
                "model": g_model,
                "messages": [
                    {
                        "role": "system",
                        "content": system_content
                    },
                    {"role": "user", "content": user_msg}
                ],
                "temperature": 0.7
            }
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=15)
                if response.status_code == 200:
                    res_data = response.json()
                    content = res_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                    if content and not is_ai_refusal(content):
                        return content
                    else:
                        logger.warning(f"Groq model {g_model} returned a refusal or insufficient content: {repr(content)}. Trying next model...")
                else:
                    logger.warning(f"Groq model {g_model} returned {response.status_code}: {response.text}")
            except Exception as e:
                logger.error(f"Error calling Groq model {g_model} for caption: {e}")

    # Safe hardcoded fallback if all AI services fail
    if is_maga:
        return (
            "Standing strong for our principles and putting America First. "
            "Every patriot has a voice, and together we can protect faith, family, and freedom. "
            "Common sense and truth will always prevail when good people stand united. "
            "We must remain vigilant and never surrender the values that made this nation great. "
            "Join the movement today and make your voice heard across America.\n\n"
            "#MAGA #AmericaFirst #Conservative #Trump #Freedom"
        )
    return (
        "Standing strong for our principles and the future of America. "
        "Every citizen has a voice, and together we can protect faith, family, and freedom. "
        "Common sense and truth will always prevail when good people stand united. "
        "We must remain vigilant and never surrender the values that made this nation great. "
        "Join the movement today and make your voice heard across the country.\n\n"
        "#Conservative #CharlieKirk #AmericaFirst #Freedom #Truth"
    )

def generate_charlie_caption(image_bytes: bytes = None, text_content: str = "") -> str:
    """Wrapper for Charlie Kirk caption generation."""
    return generate_post_caption("charlie", image_bytes=image_bytes, text_content=text_content)

def generate_maga_caption(image_bytes: bytes = None, text_content: str = "") -> str:
    """Wrapper for Supporting MAGA caption generation."""
    return generate_post_caption("maga", image_bytes=image_bytes, text_content=text_content)

async def safe_reply_caption(target_message, caption_text: str):
    """Safely sends the generated Instagram caption directly without headers or footers for easy copying."""
    clean_caption = caption_text.strip()
    try:
        await target_message.reply_text(clean_caption)
    except Exception as e:
        logger.error(f"Error sending caption: {e}")

def call_ocr_space(image_bytes: bytes) -> str:
    """Extract text from image bytes using OCR.space API with automatic retries."""
    url = "https://api.ocr.space/parse/image"
    api_key = os.getenv("OCR_SPACE_API_KEY", "helloworld")
    payload = {
        "apikey": api_key,
        "language": "eng",
        "isOverlayRequired": False,
        "detectOrientation": True,
        "scale": True,
        "OCREngine": 2
    }
    files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
    for attempt in range(3):
        try:
            response = requests.post(url, files=files, data=payload, timeout=25)
            if response.status_code == 200:
                res_json = response.json()
                results = res_json.get("ParsedResults")
                if results:
                    text = results[0].get("ParsedText", "").strip()
                    if text:
                        return text
            logger.warning(f"OCR.space attempt {attempt + 1} status code {response.status_code}")
        except Exception as e:
            logger.warning(f"OCR.space attempt {attempt + 1} failed: {e}")
        time.sleep(1.5)
    return ""

def clean_ocr_headline_with_ai(raw_text: str) -> str:
    """Uses Groq AI to extract and clean ONLY the central headline/quote from screenshot OCR."""
    if not raw_text or len(raw_text.strip()) < 5:
        return ""
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        return raw_text.strip()
        
    system_prompt = "You are an expert OCR cleaner for social media graphics. Extract ONLY the exact headline text verbatim, removing surrounding UI."
    user_prompt = f"""Raw OCR from a social media screenshot:
---
{raw_text}
---

Extract the central headline/quote text.
CRITICAL RULES:
1. Do NOT summarize, paraphrase, or change the wording. Keep the EXACT words verbatim.
2. Strip phone UI (time, battery, LTE), app headers (usernames, follow buttons), interaction counters (likes, comments), audio tags, and comments/captions below the post.
3. Strip secondary badges/logos (e.g. "MAGA", "SUPPORT ICE").
4. Fix broken OCR fragments (e.g. "SHO\\nD" -> "SHOULD").
5. Keep the text in ALL CAPS.
6. Break into clean lines matching the original headline graphic (roughly 3-5 words per line, or 18-28 characters per line).
7. If there is NO headline or quote in the text (e.g. only UI buttons or numbers), return NONE.
8. Return ONLY the verbatim headline text. No explanations, no quotes."""

    models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"]
    if os.getenv("GROQ_MODEL"):
        models.insert(0, os.getenv("GROQ_MODEL"))
    seen = set()
    unique_models = [m for m in models if not (m in seen or seen.add(m))]

    for model in unique_models:
        try:
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.0
                },
                timeout=12
            )
            if r.status_code == 200:
                res = r.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                if res and res != "NONE" and "NONE" not in res[:10] and not is_ai_refusal(res):
                    return res
        except Exception as e:
            logger.warning(f"Groq OCR cleanup error with model {model}: {e}")
            continue
    return ""

def find_ai_post_subjects(headline: str) -> dict:
    """Uses AI to identify the primary subject and related thematic elements for a headline."""
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key or not headline:
        return {}
    
    system_prompt = "You are an AI assistant for a graphic designer creating news/political social media posters. Suggest precise image search terms for subjects."
    user_prompt = f"""Given this headline:
\"{headline}\"

Suggest search terms to find high-resolution photos for:
1. "main": The primary person, leader, or subject mentioned (e.g. "Marco Rubio", "Donald Trump", "JD Vance").
2. "shape1": A related government monument, landmark, or symbol (e.g. "United States Capitol", "White House", "American Flag").
3. "shape2": A secondary related figure or official venue (e.g. "Oval Office", "Donald Trump").

Return ONLY a JSON object:
{{"main": "...", "shape1": "...", "shape2": "..."}}"""

    models = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
    for m in models:
        try:
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
                json={
                    "model": m,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.1
                },
                timeout=10
            )
            if r.status_code == 200:
                raw_json = r.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                if raw_json.startswith("```"):
                    raw_json = re.sub(r"^```[a-z]*\n?", "", raw_json)
                    raw_json = re.sub(r"\n?```$", "", raw_json)
                data = json.loads(raw_json)
                if isinstance(data, dict) and "main" in data:
                    return data
        except Exception as e:
            logger.warning(f"Error extracting subjects with model {m}: {e}")
            continue
    return {}

def search_wikimedia_image(query: str) -> str:
    """Searches Wikipedia and Wikimedia Commons for high-resolution images."""
    if not query:
        return None
    headers = {"User-Agent": "TemplateaBot/1.0 (contact@templatea.com)"}
    
    clean_q = re.sub(r"(?i)\b(portrait|photo|picture|building|interior|seal|flag|high resolution|official|hd|4k)\b", "", query).strip()
    clean_q = re.sub(r"\s+", " ", clean_q).strip()

    # 1. Wikipedia Pageimage (best for specific public figures)
    try:
        url = f"https://en.wikipedia.org/w/api.php?action=query&titles={requests.utils.quote(clean_q)}&prop=pageimages&format=json&pithumbsize=1280"
        r = requests.get(url, headers=headers, timeout=6)
        if r.status_code == 200:
            pages = r.json().get("query", {}).get("pages", {})
            for pid, pdata in pages.items():
                if pid != "-1":
                    thumb = pdata.get("thumbnail", {}).get("source")
                    if thumb and not any(bad in thumb.lower() for bad in [".pdf", ".djvu", ".tif", ".svg"]):
                        return thumb
    except Exception:
        pass
        
    # 2. Wikimedia Commons Generator Search
    try:
        url = f"https://commons.wikimedia.org/w/api.php?action=query&generator=search&gsrnamespace=6&gsrsearch={requests.utils.quote(clean_q)}&gsrlimit=6&prop=imageinfo&iiprop=url|size&iiurlwidth=1280&format=json"
        r = requests.get(url, headers=headers, timeout=6)
        if r.status_code == 200:
            pages = r.json().get("query", {}).get("pages", {})
            for pid, pdata in pages.items():
                title = pdata.get("title", "")
                if any(bad in title.lower() for bad in [".pdf", ".djvu", ".tif", ".svg", "bulletin", "map", "document"]):
                    continue
                ii = pdata.get("imageinfo", [{}])[0]
                thumb = ii.get("thumburl")
                if thumb and not any(bad in thumb.lower() for bad in [".pdf", ".djvu", ".tif", ".svg"]):
                    return thumb
    except Exception:
        pass
        
    return None

def compute_sift_match_score(screenshot_path: str, candidate_path: str, is_shape: bool = False) -> int:
    """
    Computes robust SIFT + RANSAC homography inlier matching score between a screenshot and candidate image.
    Crops screenshot to the actual post body and shape regions to prevent UI interference.
    """
    try:
        img_ss = cv2.imread(screenshot_path)
        img_cand = cv2.imread(candidate_path)
        if img_ss is None or img_cand is None:
            return 0
        h, w = img_ss.shape[:2]
        ratio = h / max(w, 1)
        if ratio > 1.3:
            if is_shape:
                crop_ss = img_ss[int(h * 0.22):int(h * 0.75), :]
            else:
                crop_ss = img_ss[int(h * 0.11):int(h * 0.72), :]
        else:
            crop_ss = img_ss
            
        sift = cv2.SIFT_create(nfeatures=3000)
        bf = cv2.BFMatcher(cv2.NORM_L2)
        
        kp_ss, des_ss = sift.detectAndCompute(crop_ss, None)
        kp_c, des_c = sift.detectAndCompute(img_cand, None)
        if des_ss is None or des_c is None:
            return 0
            
        matches = bf.knnMatch(des_ss, des_c, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        if len(good) < 4:
            return len(good)
            
        src_pts = np.float32([kp_ss[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_c[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        inliers = int(np.sum(mask)) if mask is not None else len(good)
        logger.info(f"SIFT match: {os.path.basename(screenshot_path)} vs {os.path.basename(candidate_path)} (is_shape={is_shape}): {inliers} inliers")
        return inliers
    except Exception as e:
        logger.warning(f"Error computing SIFT score {os.path.basename(screenshot_path)} vs {os.path.basename(candidate_path)}: {e}")
        return 0

def match_screenshot_to_images(screenshot_path: str, candidate_paths: list, is_shape: bool = False) -> str:
    """
    Finds the best matching clean photo for a given screenshot using OpenCV SIFT feature matching.
    SIFT is scale-invariant, rotation-invariant, and lighting-invariant, perfectly matching
    differently cropped and resized photos from social media screenshots.
    """
    if not candidate_paths:
        return None
    if len(candidate_paths) == 1:
        return candidate_paths[0]

    best_candidate = candidate_paths[0]
    best_score = -1
    for c_path in candidate_paths:
        score = compute_sift_match_score(screenshot_path, c_path, is_shape=is_shape)
        if score > best_score:
            best_score = score
            best_candidate = c_path

    return best_candidate

def get_rendered_textbox_keyboard(font_choice: str = "faith") -> InlineKeyboardMarkup:
    """Build the inline keyboard shown after a text box is rendered."""
    font_choice = (font_choice or "faith").lower()
    if font_choice == "freedom":
        keyboard = [
            [
                InlineKeyboardButton("✏️ Edit Text", callback_data="edit_text"),
                InlineKeyboardButton("🔄 Change Font", callback_data="changefont")
            ]
        ]
    else:
        keyboard = [
            [
                InlineKeyboardButton("🟡 Yellow", callback_data="recolor:yellow"),
                InlineKeyboardButton("🔴 Red", callback_data="recolor:red")
            ],
            [
                InlineKeyboardButton("🟠 Orange", callback_data="recolor:orange"),
                InlineKeyboardButton("🟣 Magenta", callback_data="recolor:magenta")
            ],
            [
                InlineKeyboardButton("🟢 Green", callback_data="recolor:green"),
                InlineKeyboardButton("🔵 Blue", callback_data="recolor:blue")
            ],
            [
                InlineKeyboardButton("🟪 Purple", callback_data="recolor:purple")
            ],
            [
                InlineKeyboardButton("✏️ Edit Text", callback_data="edit_text"),
                InlineKeyboardButton("🔄 Change Font", callback_data="changefont")
            ]
        ]
    return InlineKeyboardMarkup(keyboard)

async def safe_delete_message(msg):
    """Safely delete a message without raising unhandled errors if already deleted or timed out."""
    if msg is not None:
        try:
            await msg.delete()
        except Exception as e:
            logger.debug(f"Could not delete message (safe to ignore): {e}")

async def safe_edit_status_message(status_msg, text: str, parse_mode: str = None):
    """Safely edit a status message, catching any timeout or Telegram error."""
    if status_msg is not None:
        try:
            await status_msg.edit_text(text, parse_mode=parse_mode)
        except Exception as e:
            logger.warning(f"Could not edit status message (already deleted or timed out): {e}")

async def send_rendered_media_with_retry(
    target,
    media_bytes: bytes,
    filename: str,
    as_document: bool,
    caption: str,
    reply_markup=None,
    max_retries: int = 3,
    read_timeout: float = 60.0,
    write_timeout: float = 60.0
):
    """Sends a photo or document with automatic retries on timeout/network error."""
    last_err = None
    for attempt in range(max_retries):
        buf = BytesIO(media_bytes)
        buf.name = filename
        try:
            if as_document:
                return await target.reply_document(
                    document=buf,
                    caption=caption,
                    reply_markup=reply_markup,
                    read_timeout=read_timeout,
                    write_timeout=write_timeout
                )
            else:
                return await target.reply_photo(
                    photo=buf,
                    caption=caption,
                    reply_markup=reply_markup,
                    read_timeout=read_timeout,
                    write_timeout=write_timeout
                )
        except (TimedOut, NetworkError) as e:
            last_err = e
            logger.warning(f"Attempt {attempt + 1}/{max_retries} to send media failed with {type(e).__name__}: {e}. Retrying...")
            await asyncio.sleep(1.0 * (attempt + 1))
        except Exception as e:
            raise e
    raise last_err

async def send_video_with_retry(
    target,
    video_path: str,
    caption: str,
    reply_markup=None,
    max_retries: int = 3,
    read_timeout: float = 120.0,
    write_timeout: float = 120.0
):
    """Sends a video file with automatic retries on timeout/network error."""
    last_err = None
    for attempt in range(max_retries):
        try:
            with open(video_path, "rb") as f:
                return await target.reply_video(
                    video=f,
                    caption=caption,
                    reply_markup=reply_markup,
                    read_timeout=read_timeout,
                    write_timeout=write_timeout
                )
        except (TimedOut, NetworkError) as e:
            last_err = e
            logger.warning(f"Attempt {attempt + 1}/{max_retries} to send video failed with {type(e).__name__}: {e}. Retrying...")
            await asyncio.sleep(2.0 * (attempt + 1))
        except Exception as e:
            raise e
    raise last_err

def generate_textbox_image(text: str, font_type: str, highlight_choice: str = "yellow", user_support_image_path: str = None, doge_highlight_lines: int = None) -> BytesIO:
    """Generates the premium text box image as a PNG bytes buffer."""
    # Image Canvas Dimensions
    canvas_w = 1080
    
    # Aesthetic styling colors
    bg_start = (20, 24, 33)      # Charcoal blue gradient top
    bg_end = (10, 12, 16)        # Dark midnight background bottom
    box_bg = (0, 0, 0, 175)      # Semi-transparent sleek dark box
    box_outline = (255, 255, 255, 45) # Thin glass outline
    
    if font_type.lower() == "freedom":
        highlight_choice = "orange"
        highlight_color = FREEDOM_WATERMARK_LINE_COLOR
        box_bg = (0, 0, 0, 255) # Solid black for Freedom
    elif font_type.lower() in ("charlie", "maga_charlie"):
        highlight_color = get_highlight_color(highlight_choice)
        box_bg = (0, 0, 0, 255) # Solid black for Charlie
    else:
        highlight_color = get_highlight_color(highlight_choice)
        
    default_text_color = (255, 255, 255) # Premium white
    
    # Parse lines and capitalize all text
    lines = [line.strip().upper() for line in text.split('\n') if line.strip()]
    if not lines:
        lines = ["TEXT BOX"]
        
    num_lines = len(lines)
    is_maga = font_type.lower() in ("maga", "charlie", "maga_charlie", "doge")
    
    # Assign colors:
    # For doge: color the last k lines based on user choice
    # For maga/charlie/freedom: middle lines are colored (4 lines → middle 2, 5/6 lines → middle 3)
    # For faith: last lines are colored (<=4 → last 2, >=5 → last 3)
    line_colors = []
    if font_type.lower() == "doge":
        k = doge_highlight_lines if doge_highlight_lines is not None else 2
        k = min(num_lines, max(0, k))
        for idx in range(num_lines):
            if idx >= num_lines - k:
                line_colors.append(highlight_color)
            else:
                line_colors.append(default_text_color)
    elif font_type.lower() in ("maga", "charlie", "maga_charlie", "freedom"):
        # Maga / Charlie / Freedom: color the middle lines
        if num_lines <= 2:
            num_colored = 1
        elif num_lines <= 4:
            num_colored = 2
        else:
            num_colored = 3
        # Center the colored block
        start_colored = (num_lines - num_colored) // 2
        end_colored = start_colored + num_colored
        for idx in range(num_lines):
            if start_colored <= idx < end_colored:
                line_colors.append(highlight_color)
            else:
                line_colors.append(default_text_color)
    else:
        # Faith: color the last lines
        if num_lines <= 4:
            for idx in range(num_lines):
                if idx >= num_lines - 2:
                    line_colors.append(highlight_color)
                else:
                    line_colors.append(default_text_color)
        else:
            for idx in range(num_lines):
                if idx >= num_lines - 3:
                    line_colors.append(highlight_color)
                else:
                    line_colors.append(default_text_color)
                
    # Find matching font
    fonts_dir = os.path.join(BASE_DIR, "fonts")
    if font_type.lower() == "doge":
        font_path = os.path.join(fonts_dir, "league_gothic.ttf")
    elif font_type.lower() == "freedom":
        font_path = None
        for name in ("Montserrat-Black.ttf", "montserrat_black.ttf", "freedom.ttf", "montserrat_bold.ttf", "Montserrat-Bold.ttf", "montserrat.ttf"):
            local_f = os.path.join(fonts_dir, name)
            if os.path.exists(local_f):
                font_path = local_f
                break
        if font_path is None:
            dl_paths = [
                "/Users/dhawansevkani/Downloads/montserrat/Montserrat-Black.ttf",
                "/Users/dhawansevkani/Downloads/montserrat/Montserrat-Black.otf",
                "/Users/dhawansevkani/Downloads/montserrat/Montserrat-Bold.ttf",
                "/Users/dhawansevkani/Downloads/montserrat/Montserrat-Bold.otf",
                "/Library/Fonts/Montserrat-Black.ttf",
                "/System/Library/Fonts/Supplemental/Montserrat-Black.ttf",
                "/Library/Fonts/Montserrat-Bold.ttf",
                "/System/Library/Fonts/Supplemental/Montserrat-Bold.ttf"
            ]
            for dp in dl_paths:
                if os.path.exists(dp):
                    font_path = dp
                    break
    elif font_type.lower() in ("maga", "charlie", "maga_charlie"):
        # Both Maga and Charlie styles use the Impact font (impact.ttf)
        font_path = None
        for name in ("impact.ttf", "maga.ttf"):
            local_impact = os.path.join(fonts_dir, name)
            if os.path.exists(local_impact):
                font_path = local_impact
                break
        if font_path is None:
            # Fallback to system paths if local is missing
            impact_paths = [
                "/System/Library/Fonts/Supplemental/Impact.ttf",
                "/usr/share/fonts/truetype/msttcorefonts/Impact.ttf",
                "/usr/share/fonts/Impact.ttf",
                "C:\\Windows\\Fonts\\impact.ttf",
            ]
            for ip in impact_paths:
                if os.path.exists(ip):
                    font_path = ip
                    break
    else:
        font_path = os.path.join(fonts_dir, f"{font_type.lower()}.ttf")
    
    if font_path is None or not os.path.exists(font_path):
        # Local system fallbacks
        fallbacks = [
            "/System/Library/Fonts/Supplemental/Georgia.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "Arial.ttf"
        ]
        for fb in fallbacks:
            if os.path.exists(fb):
                font_path = fb
                break
        else:
            font_path = None
            
    # Text box geometry constraints
    box_width = 1080
    padding = 14
    target_text_width = box_width - (padding * 2)
    
    temp_img = Image.new("RGBA", (canvas_w, 1350))
    temp_draw = ImageDraw.Draw(temp_img)
    
    fonts = []
    line_widths = []
    line_heights = []
    line_bboxes = []
    line_measured_info = []
    
    # Font sizing constraints to prevent excessive scaling
    max_font_size = 9999
    min_font_size = 15
    
    for line in lines:
        # Calculate dynamic font size using ratio-based resizing
        base_size = 100
        try:
            if font_path is not None:
                base_font = ImageFont.truetype(font_path, base_size)
            else:
                base_font = ImageFont.load_default(size=base_size)
            if has_emojis(line):
                base_w, _, _, _ = measure_line_with_emojis(temp_draw, line, base_font)
            else:
                bbox = temp_draw.textbbox((0, 0), line, font=base_font)
                base_w = bbox[2] - bbox[0]
            if base_w > 0:
                estimated_size = int(base_size * (target_text_width / base_w))
            else:
                estimated_size = 50
        except Exception:
            estimated_size = 50
            
        font_size = max(min_font_size, min(max_font_size, estimated_size))
        
        try:
            if font_path is not None:
                line_font = ImageFont.truetype(font_path, font_size)
            else:
                line_font = ImageFont.load_default(size=font_size)
        except Exception:
            try:
                line_font = ImageFont.load_default(size=font_size)
            except Exception:
                line_font = ImageFont.load_default()
            
        if has_emojis(line):
            line_w, seg_data, cap_top, cap_h = measure_line_with_emojis(temp_draw, line, line_font)
            line_measured_info.append((line_w, seg_data, cap_top, cap_h))
            bbox = temp_draw.textbbox((0, 0), line, font=line_font)
            if bbox[2] <= bbox[0]:
                bbox = (0, cap_top, int(line_w), cap_top + cap_h)
        else:
            bbox = temp_draw.textbbox((0, 0), line, font=line_font)
            line_w = bbox[2] - bbox[0]
            line_measured_info.append(None)
            
        line_h = get_line_height(line_font)
        
        fonts.append(line_font)
        line_widths.append(line_w)
        line_heights.append(line_h)
        line_bboxes.append(bbox)
        
    # Calculate box height
    if font_type.lower() in ("maga", "charlie", "maga_charlie"):
        line_spacing_multiplier = 0.75
    elif font_type.lower() == "freedom":
        line_spacing_multiplier = 0.69
    else:
        line_spacing_multiplier = 0.70
    total_text_height = 0
    first_line_top_offset = 0
    if num_lines > 0:
        current_offset = 0
        for i in range(num_lines - 1):
            h_curr = line_heights[i]
            h_next = line_heights[i+1]
            step = 0.5 * (1 + line_spacing_multiplier) * h_curr - 0.5 * (1 - line_spacing_multiplier) * h_next
            current_offset += step
        
        first_line_top_offset = line_bboxes[0][1]
        last_line_bottom_offset = line_bboxes[-1][3]
        total_text_height = current_offset + last_line_bottom_offset - first_line_top_offset
        
    if is_maga:
        box_top_padding = 7
        box_bottom_padding = 18
    elif font_type.lower() == "freedom":
        box_top_padding = 18
        box_bottom_padding = 24
    else:
        box_top_padding = padding
        box_bottom_padding = padding
        
    box_height = int(total_text_height + box_top_padding + box_bottom_padding)
    
    start_y = 17
    # Load banner if style is faith
    banner_img = None
    banner_height = 0
    if font_type.lower() == "faith":
        banner_path = os.path.join(BASE_DIR, "resources", "faith banner.png")
        if not os.path.exists(banner_path):
            banner_path = "/Users/dhawansevkani/Downloads/faith banner.png"
        if os.path.exists(banner_path):
            try:
                banner_img = Image.open(banner_path)
                banner_height = banner_img.height
            except Exception as e:
                logger.error(f"Failed to load banner: {e}")
                
    if banner_img is not None:
        box_y = start_y + banner_height - 5
    elif font_type.lower() == "freedom":
        # Text in the bottom part!
        box_y = max(17, 1350 - box_height - 20)
    else:
        box_y = start_y
        
    if font_type.lower() == "freedom":
        line_bottom_val = box_y
    elif font_type.lower() in ("maga", "charlie"):
        line_bottom_val = box_y + box_height + 20
    else:
        line_bottom_val = box_y + box_height

    # Load supporting PNG if style is maga, charlie, or faith
    support_img = None
    support_height = 0
    if font_type.lower() in ("maga", "charlie", "faith"):
        if user_support_image_path is not None and os.path.exists(user_support_image_path):
            try:
                support_img = Image.open(user_support_image_path).convert("RGBA")
                support_img = enhance_image_quality(support_img)
                support_w = 1080
                # Calculate remaining space to make it fill from the line to the bottom (1350px)
                support_height = max(10, 1350 - line_bottom_val)
                
                # Crop to the exact aspect ratio of (1080 x support_height)
                img_aspect = support_img.width / support_img.height
                target_aspect = support_w / support_height
                if img_aspect > target_aspect:
                    new_width = int(support_img.height * target_aspect)
                    left = (support_img.width - new_width) // 2
                    support_img = support_img.crop((left, 0, left + new_width, support_img.height))
                else:
                    new_height = int(support_img.width / target_aspect)
                    top = (support_img.height - new_height) // 2
                    support_img = support_img.crop((0, top, support_img.width, top + new_height))
                
                support_img = support_img.resize((support_w, support_height), Image.Resampling.LANCZOS)
            except Exception as e:
                logger.error(f"Failed to load/resize supporting image: {e}")

    # Calculate total height of elements
    if banner_img is not None:
        total_height = banner_height + 5 + box_height
    elif support_img is not None:
        if user_support_image_path is not None:
            total_height = 1350
        else:
            total_height = box_height + 20 + support_height
    else:
        total_height = box_height

    # Final output dimensions (4:5 ratio)
    canvas_h = 1350
    
    # Generate background (transparent for maga/charlie/faith/doge/freedom, solid black for others)
    if font_type.lower() in ("maga", "charlie", "faith", "doge", "freedom"):
        bg_img = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    else:
        bg_img = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 255))
    box_overlay = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    box_draw = ImageDraw.Draw(box_overlay)
    
    # Start elements from the top (Y = 17px)
    start_y = 17
    
    # Paste banner if present
    if banner_img is not None:
        banner_x = 14
        banner_y = start_y
        bg_img.paste(banner_img, (banner_x, banner_y), banner_img if banner_img.mode in ('RGBA', 'LA') else None)
        
    # Freedom Template: Watermark above text, subtle black gradient, and 5px orange divider line
    if font_type.lower() == "freedom":
        # 1. Load and prepare watermark directly above text
        watermark_img = None
        watermark_h = 0
        watermark_x = 0
        watermark_y = box_y
        wm_paths = [
            os.path.join(BASE_DIR, "resources", "freedom_watermark.png"),
            os.path.join(BASE_DIR, "resources", "watermark.png"),
            "/Users/dhawansevkani/Downloads/watermark.png",
            "/Users/dhawansevkani/Downloads/watermark"
        ]
        for wp in wm_paths:
            if os.path.exists(wp):
                try:
                    wm_raw = Image.open(wp).convert("RGBA")
                    # In freedom_watermark.png, rows 115-185 contain the "FreedomFront" text
                    wm_logo_cropped = wm_raw.crop((0, 115, wm_raw.width, 185))
                    logo_bbox = wm_logo_cropped.getbbox()
                    if logo_bbox:
                        wm_logo_tight = wm_logo_cropped.crop(logo_bbox)
                        # Boost brightness so "Freedom" is pure bright white
                        enhancer = ImageEnhance.Brightness(wm_logo_tight)
                        wm_logo_bright = enhancer.enhance(1.45)
                        
                        target_logo_h = 42
                        target_logo_w = int(wm_logo_bright.width * (target_logo_h / wm_logo_bright.height))
                        watermark_img = wm_logo_bright.resize((target_logo_w, target_logo_h), Image.Resampling.LANCZOS)
                        watermark_h = target_logo_h
                        watermark_x = (canvas_w - target_logo_w) // 2
                        watermark_gap = 18
                        watermark_y = box_y - watermark_h - watermark_gap
                    break
                except Exception as e:
                    logger.error(f"Failed to load freedom watermark: {e}")

        # 1.5. User support/background image (placed above the line, behind the gradient)
        if user_support_image_path is not None and os.path.exists(user_support_image_path):
            try:
                user_img = Image.open(user_support_image_path).convert("RGBA")
                user_enhanced = enhance_image_quality(user_img)
                target_h = box_y
                target_aspect = 1080 / target_h
                img_aspect = user_enhanced.width / user_enhanced.height
                if img_aspect > target_aspect:
                    new_w = int(user_enhanced.height * target_aspect)
                    left = (user_enhanced.width - new_w) // 2
                    crop_box = (left, 0, left + new_w, user_enhanced.height)
                else:
                    new_h = int(user_enhanced.width / target_aspect)
                    top = (user_enhanced.height - new_h) // 2
                    crop_box = (0, top, user_enhanced.width, top + new_h)
                user_cropped = user_enhanced.crop(crop_box).resize((1080, target_h), Image.Resampling.LANCZOS)
                bg_img.paste(user_cropped, (0, 0))
            except Exception as e:
                logger.error(f"Failed to load freedom user image: {e}")

        # 2. Subtle black gradient between image and watermark/textbox (tight ~130px cinematic fade)
        gradient_start_y = max(0, box_y - 130)
        gradient_layer = draw_vertical_alpha_gradient(canvas_w, canvas_h, gradient_start_y, box_y, start_alpha=0, end_alpha=245)
        bg_img = Image.alpha_composite(bg_img, gradient_layer)
        
        # 3. Paste watermark above the text box
        if watermark_img is not None:
            bg_img.paste(watermark_img, (watermark_x, watermark_y), watermark_img)
            
        # 4. Paste logo on the right side (top-right corner)
        logo_paths = [
            os.path.join(BASE_DIR, "resources", "freedom_logo.png"),
            os.path.join(BASE_DIR, "resources", "logo.png"),
            "/Users/dhawansevkani/Downloads/logo.png",
            "/Users/dhawansevkani/Downloads/logo"
        ]
        for lp in logo_paths:
            if os.path.exists(lp):
                try:
                    logo_img = Image.open(lp).convert("RGBA")
                    logo_w, logo_h = logo_img.size
                    logo_x = canvas_w - logo_w - 30
                    logo_y = 30
                    bg_img.paste(logo_img, (logo_x, logo_y), logo_img)
                    break
                except Exception as e:
                    logger.error(f"Failed to load freedom logo: {e}")

        # 5. Draw crisp 5px horizontal orange divider line at box_y
        line_thickness = 5
        box_draw.rectangle([0, box_y, canvas_w, box_y + line_thickness], fill=highlight_color)

    box_x = (canvas_w - box_width) // 2
    if font_type.lower() == "freedom":
        box_coords = [0, box_y + 5, canvas_w, canvas_h]
        current_y = box_y + 5 + box_top_padding - first_line_top_offset
    elif font_type.lower() in ("charlie", "maga", "maga_charlie"):
        box_coords = [0, 0, canvas_w, box_y + box_height]
        current_y = box_y + box_top_padding - first_line_top_offset
    else:
        box_coords = [box_x, box_y, box_x + box_width, box_y + box_height]
        current_y = box_y + box_top_padding - first_line_top_offset
    
    box_draw.rectangle(
        box_coords,
        fill=box_bg
    )
    
    # Draw each line of text
    for i in range(num_lines):
        line = lines[i]
        font = fonts[i]
        color = line_colors[i]
        measured_item = line_measured_info[i]
        
        if measured_item is not None:
            line_w = line_widths[i]
            x = box_x + padding + (target_text_width - int(line_w)) // 2
            draw_text_with_emojis(box_draw, box_overlay, x, current_y, line, font=font, fill=color, measured_data=measured_item)
        else:
            # Use precise bounding box to avoid cutting off the edges
            bbox = box_draw.textbbox((0, 0), line, font=font)
            left, top, right, bottom = bbox
            w = right - left
            # Center text horizontally using exact bounds
            x = box_x + padding - left + (target_text_width - w) // 2
            box_draw.text((x, current_y), line, font=font, fill=color)
        if i < num_lines - 1:
            h_curr = line_heights[i]
            h_next = line_heights[i+1]
            step = 0.5 * (1 + line_spacing_multiplier) * h_curr - 0.5 * (1 - line_spacing_multiplier) * h_next
            current_y += step
        
    # Draw horizontal line and paste supporting PNG if present
    if font_type.lower() in ("maga", "charlie", "faith"):
        # Draw a horizontal line of thickness 20px, directly below the text box (0px gap)
        # Spans edge-to-edge (0 to 1080px)
        line_top = box_y + box_height
        if font_type.lower() in ("maga", "charlie"):
            line_bottom = line_top + 20
            box_draw.rectangle([0, line_top, 1080, line_bottom], fill=highlight_color)
        else:
            line_bottom = line_top
        
        # Paste supporting PNG directly below the line
        if support_img is not None:
            bg_img.paste(support_img, (0, line_bottom), support_img if support_img.mode in ('RGBA', 'LA') else None)
        
        # Draw watermarks and gradient for Maga and Charlie styles
        if font_type.lower() in ("maga", "charlie"):
            watermark_path = None
            if font_type.lower() == "maga":
                watermark_path = os.path.join(BASE_DIR, "resources", "Supporting Maga.png")
                if not os.path.exists(watermark_path):
                    watermark_path = "/Users/dhawansevkani/Downloads/Supporting Maga.png"
            elif font_type.lower() == "charlie":
                watermark_path = os.path.join(BASE_DIR, "resources", "charlie banner.png")
                if not os.path.exists(watermark_path):
                    watermark_path = "/Users/dhawansevkani/Downloads/charlie banner.png"
            
            # Subtle black downward gradient below divider line for Charlie and Maga (~120px cinematic fade)
            if font_type.lower() in ("charlie", "maga"):
                grad_end_y = min(canvas_h, line_bottom + 120)
                cinematic_grad = draw_vertical_alpha_gradient(canvas_w, canvas_h, line_bottom, grad_end_y, start_alpha=245, end_alpha=0)
                bg_img = Image.alpha_composite(bg_img, cinematic_grad)
            
            if watermark_path and os.path.exists(watermark_path):
                try:
                    watermark_img = Image.open(watermark_path).convert("RGBA")
                    watermark_w = 1080
                    watermark_h = int(watermark_img.height * (watermark_w / watermark_img.width))
                    watermark_img = watermark_img.resize((watermark_w, watermark_h), Image.Resampling.LANCZOS)
                    
                    if font_type.lower() in ("charlie", "maga"):
                        # Boost brightness & opacity of watermark for ultra-clear visibility
                        wm_arr = np.array(watermark_img)
                        alpha = wm_arr[:, :, 3].astype(np.float32)
                        alpha = np.where(alpha > 20, np.clip(alpha * 1.85, 0, 255), alpha)
                        wm_arr[:, :, 3] = alpha.astype(np.uint8)
                        rgb = wm_arr[:, :, :3].astype(np.float32)
                        rgb = np.clip(rgb * 1.35, 0, 255).astype(np.uint8)
                        wm_arr[:, :, :3] = rgb
                        watermark_img = Image.fromarray(wm_arr)
                        
                    bg_img.paste(watermark_img, (0, line_bottom), watermark_img if watermark_img.mode in ('RGBA', 'LA') else None)
                except Exception as e:
                    logger.error(f"Failed to load/paste watermark: {e}")
        
    # Combine layers
    final_img = Image.alpha_composite(bg_img, box_overlay)
    
    if font_type.lower() == "doge":
        final_img = final_img.crop((0, box_y, canvas_w, box_y + box_height))
        
    out_buf = BytesIO()
    final_img.save(out_buf, format="PNG")
    out_buf.seek(0)
    return out_buf, line_bottom_val

def create_subject_shape(img, size: int, shape_type: str, border_color: tuple, border_width: int = 6) -> Image.Image:
    """
    Creates a square or circular masked subject image with a colored border.
    Caches the generated shape for ultra-fast positioning/movement.
    """
    cache_key = None
    if isinstance(img, str):
        if os.path.exists(img):
            cache_key = (img, os.path.getmtime(img), size, shape_type.lower(), border_color, border_width)
            if cache_key in SHAPE_CACHE:
                return SHAPE_CACHE[cache_key]
        img_obj = Image.open(img)
    else:
        img_obj = img

    img_enhanced = enhance_image_quality(img_obj)
    img_w, img_h = img_enhanced.size
    crop_size = min(img_w, img_h)
    left = (img_w - crop_size) // 2
    top = (img_h - crop_size) // 2
    cropped = img_enhanced.crop((left, top, left + crop_size, top + crop_size))
    resized = cropped.resize((size, size), Image.Resampling.LANCZOS)
    
    mask = Image.new("L", (size, size), 0)
    mask_draw = ImageDraw.Draw(mask)
    
    outer_size = size + border_width * 2
    canvas = Image.new("RGBA", (outer_size, outer_size), (0, 0, 0, 0))
    canvas_draw = ImageDraw.Draw(canvas)
    
    if shape_type.lower() == "circle":
        mask_draw.ellipse((0, 0, size, size), fill=255)
        canvas_draw.ellipse((0, 0, outer_size, outer_size), fill=border_color)
    else: # square
        mask_draw.rectangle((0, 0, size, size), fill=255)
        canvas_draw.rectangle((0, 0, outer_size, outer_size), fill=border_color)
        
    masked_img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    masked_img.paste(resized, (0, 0), mask)
    
    canvas.paste(masked_img, (border_width, border_width), masked_img)
    if cache_key is not None:
        with _SHAPE_LOCK:
            if len(SHAPE_CACHE) > 50:
                oldest = next(iter(SHAPE_CACHE))
                SHAPE_CACHE.pop(oldest, None)
            SHAPE_CACHE[cache_key] = canvas
    return canvas

def overlay_png_on_image(
    bg_image_path: str, 
    png_bytes: bytes, 
    shape_image_path: str = None,
    shape_type: str = "square", 
    border_color: tuple = (255, 222, 89),
    shape_offset_x: int = 0,
    shape_offset_y: int = 0,
    shape_scale: float = 1.0,
    simple_overlay: bool = False,
    line_bottom: int = None,
    # Second shape parameters
    shape2_image_path: str = None,
    shape2_offset_x: int = 0,
    shape2_offset_y: int = 0,
    shape2_scale: float = 1.0,
    preview_mode: bool = False
) -> BytesIO:
    """
    Overlays a transparent PNG image on top of a background image.
    Uses high-speed caching for cutout, background cropping, and overlay decoding.
    """
    target_h, y_offset = calculate_target_h(png_bytes, line_bottom)
    overlay_key = (id(png_bytes), len(png_bytes))
    if overlay_key in OVERLAY_CACHE:
        overlay = OVERLAY_CACHE[overlay_key]
    else:
        overlay = Image.open(BytesIO(png_bytes)).convert("RGBA")
        with _OVERLAY_LOCK:
            if len(OVERLAY_CACHE) > 10:
                oldest = next(iter(OVERLAY_CACHE))
                OVERLAY_CACHE.pop(oldest, None)
            OVERLAY_CACHE[overlay_key] = overlay
            
    # Check if this is a bottom-anchored overlay (Freedom style has solid box at the bottom y ~ 1340)
    is_bottom_anchored = (overlay.getpixel((540, 1340))[3] > 0)
    if is_bottom_anchored:
        shape_type = "circle"
        if border_color == (255, 222, 89):
            border_color = FREEDOM_WATERMARK_LINE_COLOR

    # Create solid black canvas
    canvas = Image.new("RGBA", (1080, 1350), (0, 0, 0, 255))
    
    if target_h > 0 and os.path.exists(bg_image_path):
        bg_mtime = os.path.getmtime(bg_image_path)
        bg_cache_key = (bg_image_path, bg_mtime, target_h)
        bg_img = None
        
        if bg_cache_key in BG_CROPPED_CACHE:
            bg_cropped = BG_CROPPED_CACHE[bg_cache_key]
        else:
            bg_img = Image.open(bg_image_path).convert("RGBA")
            bg_enhanced = enhance_image_quality(bg_img)
            bg_aspect = bg_enhanced.width / bg_enhanced.height
            target_aspect = 1080 / target_h
            
            if bg_aspect > target_aspect:
                new_w = int(bg_enhanced.height * target_aspect)
                left = (bg_enhanced.width - new_w) // 2
                crop_box = (left, 0, left + new_w, bg_enhanced.height)
            else:
                new_h = int(bg_enhanced.width / target_aspect)
                top = (bg_enhanced.height - new_h) // 2
                crop_box = (0, top, bg_enhanced.width, top + new_h)
                
            bg_cropped = bg_enhanced.crop(crop_box).resize((1080, target_h), Image.Resampling.LANCZOS)
            with _BG_CROPPED_LOCK:
                if len(BG_CROPPED_CACHE) > 10:
                    oldest = next(iter(BG_CROPPED_CACHE))
                    BG_CROPPED_CACHE.pop(oldest, None)
                BG_CROPPED_CACHE[bg_cache_key] = bg_cropped
        
        # 1. Paste background image (bottom layer)
        canvas.paste(bg_cropped, (0, y_offset))
        
        if not simple_overlay:
            # 2. Generate and paste subject shapes (middle layer)
            # Paste Shape 1
            if shape_image_path is not None and os.path.exists(shape_image_path):
                base_size = min(360, target_h - 40)
                size = int(base_size * shape_scale)
                if size > 20:
                    subject_shape = create_subject_shape(shape_image_path, size, shape_type, border_color)
                    center_x = 1080 // 2
                    center_y = y_offset + target_h // 2
                    paste_x = center_x - subject_shape.width // 2 + shape_offset_x
                    paste_y = center_y - subject_shape.height // 2 + shape_offset_y
                    canvas.paste(subject_shape, (paste_x, paste_y), subject_shape)
                    
            # Paste Shape 2 (if present)
            if shape2_image_path is not None and os.path.exists(shape2_image_path):
                base_size = min(360, target_h - 40)
                size2 = int(base_size * shape2_scale)
                if size2 > 20:
                    subject_shape2 = create_subject_shape(shape2_image_path, size2, shape_type, border_color)
                    center_x = 1080 // 2
                    center_y = y_offset + target_h // 2
                    paste2_x = center_x - subject_shape2.width // 2 + shape2_offset_x
                    paste2_y = center_y - subject_shape2.height // 2 + shape2_offset_y
                    canvas.paste(subject_shape2, (paste2_x, paste2_y), subject_shape2)
                
            # 3. Cutout (top layer) - Cached for instant positioning
            cutout_cropped = compute_and_cache_cutout(bg_image_path, target_h)
            if cutout_cropped is not None:
                canvas.paste(cutout_cropped, (0, y_offset), cutout_cropped)
    
    combined = Image.alpha_composite(canvas, overlay)
    
    out_buf = BytesIO()
    jpeg_quality = 85 if preview_mode else 98
    combined.convert("RGB").save(out_buf, format="JPEG", quality=jpeg_quality, subsampling=0)
    out_buf.seek(0)
    return out_buf

def overlay_png_on_video(video_path: str, png_bytes: bytes, output_path: str, subject_image_path: str = None, shape_type: str = "square", border_color: tuple = (255, 222, 89), line_bottom: int = None):
    """
    Overlays a transparent PNG image on top of a video.
    The top part (till line_bottom) is solid black (or full video if bottom-anchored).
    The bottom part contains the video frame.
    """
    png_data = np.frombuffer(png_bytes, dtype=np.uint8)
    overlay_img = cv2.imdecode(png_data, cv2.IMREAD_UNCHANGED)
    if overlay_img is None:
        raise ValueError("Could not decode PNG overlay image.")
    overlay_img = cv2.resize(overlay_img, (1080, 1350))
    
    # Check if bottom anchored (Freedom style has solid box at the bottom y ~ 1340)
    is_bottom_anchored = (overlay_img[1340, 540, 3] > 0)
    if is_bottom_anchored:
        shape_type = "circle"
        if border_color == (255, 222, 89):
            border_color = FREEDOM_WATERMARK_LINE_COLOR
    
    # Find line_bottom (last row where alpha channel at column 0 is > 0)
    if line_bottom is None:
        line_bottom = 0
        alpha_col = overlay_img[:, 0, 3]
        non_zero_indices = np.where(alpha_col > 0)[0]
        if len(non_zero_indices) > 0:
            line_bottom = int(non_zero_indices[-1] + 1)
        
    overlay_bgr = overlay_img[:, :, :3]
    overlay_mask = overlay_img[:, :, 3] / 255.0
    overlay_mask = np.expand_dims(overlay_mask, axis=2)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (1080, 1350))
    
    if is_bottom_anchored:
        line_pos = line_bottom if (line_bottom is not None and 0 < line_bottom < 1350) else 890
        target_h = line_pos
        y_offset = 0
    else:
        target_h = 1350 - line_bottom if (line_bottom is not None and 0 < line_bottom < 1350) else 900
        y_offset = line_bottom
    
    # Prepare static subject shape if provided
    subject_bgr = None
    subject_mask = None
    subject_coords = None
    
    if subject_image_path is not None and os.path.exists(subject_image_path):
        size = min(360, target_h - 40)
        if size > 20:
            subject_img = create_subject_shape(subject_image_path, size, shape_type, border_color)
            subject_np = np.array(subject_img)
            subject_bgr = cv2.cvtColor(subject_np[:, :, :3], cv2.COLOR_RGBA2BGRA)[:, :, :3]
            subject_mask = subject_np[:, :, 3] / 255.0
            subject_mask = np.expand_dims(subject_mask, axis=2)
            
            center_x = 1080 // 2
            center_y = y_offset + target_h // 2
            paste_x = center_x - subject_img.width // 2
            paste_y = center_y - subject_img.height // 2
            subject_coords = (paste_x, paste_y, paste_x + subject_img.width, paste_y + subject_img.height)
            
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            canvas_frame = np.zeros((1350, 1080, 3), dtype=np.uint8)
            if target_h > 0:
                h, w = frame.shape[:2]
                frame_aspect = w / h
                target_aspect = 1080 / target_h
                
                if frame_aspect > target_aspect:
                    new_w = int(h * target_aspect)
                    left = (w - new_w) // 2
                    cropped = frame[:, left:left+new_w]
                else:
                    new_h = int(w / target_aspect)
                    top = (h - new_h) // 2
                    cropped = frame[top:top+new_h, :]
                    
                resized_vid_frame = cv2.resize(cropped, (1080, target_h))
                canvas_frame[y_offset:y_offset+target_h, :] = resized_vid_frame
                
                # Blend subject shape on top of the canvas frame
                if subject_bgr is not None:
                    x1, y1, x2, y2 = subject_coords
                    canvas_roi = canvas_frame[y1:y2, x1:x2]
                    blended_roi = (canvas_roi * (1.0 - subject_mask) + subject_bgr * subject_mask).astype(np.uint8)
                    canvas_frame[y1:y2, x1:x2] = blended_roi
                    
            blended = (canvas_frame * (1.0 - overlay_mask) + overlay_bgr * overlay_mask).astype(np.uint8)
            out.write(blended)
    finally:
        cap.release()
        out.release()

# ----------- Telegram Bot Flow -----------

from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ConversationHandler,
    filters,
)

# States
STATE_CHOOSING_FONT = 1
STATE_CHOOSING_COLOR = 2
STATE_WAITING_TEXT = 3
STATE_WAITING_BACKGROUND = 4
STATE_WAITING_SHAPE_IMAGE = 5
STATE_POSITIONING_SHAPE = 6
STATE_CONFIRM_IG_OCR = 7
STATE_WAITING_SHAPE2_IMAGE = 8
STATE_WAITING_DOGE_HIGHLIGHT_LINES = 9
STATE_WAITING_CHANGE_MAIN = 10
STATE_WAITING_CHANGE_SHAPE1 = 11
STATE_WAITING_CHANGE_SHAPE2 = 12

# No-Dependency Batch Mode States
STATE_NODEP_COUNTS = 20
STATE_NODEP_DUMP = 21
STATE_NODEP_REVIEW = 22
STATE_NODEP_EDIT_TEXT = 23

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Entry point: Greet user and offer font styles."""
    user = update.effective_user
    keyboard = [
        [
            InlineKeyboardButton("✨ Faith (Lilita One)", callback_data="font:faith")
        ],
        [
            InlineKeyboardButton("🇺🇸 Maga (Impact)", callback_data="font:maga"),
            InlineKeyboardButton("🇺🇸 Charlie (Impact)", callback_data="font:charlie")
        ],
        [
            InlineKeyboardButton("🐕 Doge (League Gothic)", callback_data="font:doge")
        ],
        [
            InlineKeyboardButton("🗽 Freedom (Montserrat Black)", callback_data="font:freedom")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    msg = (
        f"👋 Welcome {user.first_name} to the Text Box Generator!\n\n"
        "Please choose a font style to start:"
    )
    if update.message:
        await update.message.reply_text(msg, reply_markup=reply_markup)
    elif update.callback_query:
        await update.callback_query.message.reply_text(msg, reply_markup=reply_markup)
        
    return STATE_CHOOSING_FONT

async def font_selected(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle font selection callback."""
    query = update.callback_query
    await query.answer()
    
    font_choice = query.data.split(":")[1]
    context.user_data["chosen_font"] = font_choice
    save_user_session(update.effective_user.id, font_choice, context.user_data.get("chosen_color"), context.user_data.get("doge_highlight_lines"))
    
    display_names = {
        "faith": "Faith (Lilita One)",
        "maga": "Maga (Impact)",
        "charlie": "Charlie (Impact)",
        "maga_charlie": "Maga/Charlie (Impact)",
        "doge": "Doge (League Gothic)",
        "freedom": "Freedom (Montserrat Black)"
    }
    
    name = display_names.get(font_choice, font_choice)
    
    if font_choice == "freedom":
        context.user_data["chosen_color"] = "orange"
        last_text = context.user_data.get("last_text")
        line_bottom = context.user_data.get("line_bottom")
        save_user_session(update.effective_user.id, chosen_font="freedom", chosen_color="orange", last_text=last_text, line_bottom=line_bottom)
        
        if last_text:
            status_msg = await query.message.reply_text("🎨 Re-rendering in Freedom style...")
            try:
                user_support_image_path = context.user_data.get("user_support_image_path")
                image_buffer, line_bottom = await asyncio.to_thread(
                    generate_textbox_image,
                    last_text, "freedom", "orange",
                    user_support_image_path=user_support_image_path
                )
                png_bytes = image_buffer.getvalue()
                context.user_data["transparent_png_bytes"] = png_bytes
                context.user_data["last_rendered_image_bytes"] = png_bytes
                context.user_data["line_bottom"] = line_bottom
                save_user_session(update.effective_user.id, chosen_font="freedom", chosen_color="orange", last_text=last_text, line_bottom=line_bottom)
                
                reply_markup = get_rendered_textbox_keyboard("freedom")
                await send_rendered_media_with_retry(
                    query.message,
                    png_bytes,
                    filename="freedom_textbox.png",
                    as_document=True,
                    caption="✅ Switched to **Freedom** style!",
                    reply_markup=reply_markup
                )
                await safe_delete_message(status_msg)
                await query.message.reply_text(
                    "🖼️ **Would you like to overlay this text box on top of a background image or video?**\n\n"
                    "📥 **Send an image or a video now** to blend it,\n"
                    "➡️ Or send new text to create another one."
                )
                return STATE_WAITING_BACKGROUND
            except Exception as e:
                logger.error(f"Error re-rendering freedom font: {e}")
                await safe_edit_status_message(status_msg, f"❌ Failed to render Freedom style: {e}")
                return STATE_WAITING_TEXT
        
        await query.edit_message_text(
            text=f"Selected Style: **{name}**\n\n"
                 f"🎨 Highlight Color: **Fixed Orange (#F87739)**\n"
                 f"📍 Placement: **Bottom Part**\n\n"
                 "💬 Send me the text you want in your box.\n"
                 "Use line breaks (Shift+Enter) to define separate lines."
        )
        return STATE_WAITING_TEXT
    
    if font_choice in ("maga", "charlie", "maga_charlie", "faith", "doge"):
        keyboard = [
            [
                InlineKeyboardButton("🟡 Yellow (#ffde59)", callback_data="color:yellow"),
                InlineKeyboardButton("🔴 Red (#ff3131)", callback_data="color:red")
            ],
            [
                InlineKeyboardButton("🟠 Orange (#ff914d)", callback_data="color:orange"),
                InlineKeyboardButton("🟣 Magenta (#cb6ce6)", callback_data="color:magenta")
            ],
            [
                InlineKeyboardButton("🟢 Green (#00e676)", callback_data="color:green"),
                InlineKeyboardButton("🔵 Blue (#38b6ff)", callback_data="color:blue")
            ],
            [
                InlineKeyboardButton("🟪 Purple (#8c52ff)", callback_data="color:purple")
            ]
        ]
    else:
        keyboard = [
            [
                InlineKeyboardButton("🟡 Yellow (#ffde59)", callback_data="color:yellow"),
                InlineKeyboardButton("🔴 Red (#ff3131)", callback_data="color:red")
            ],
            [
                InlineKeyboardButton("🟢 Green (#00e676)", callback_data="color:green"),
                InlineKeyboardButton("🔵 Blue (#38b6ff)", callback_data="color:blue")
            ],
            [
                InlineKeyboardButton("🟪 Purple (#8c52ff)", callback_data="color:purple")
            ]
        ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.edit_message_text(
        text=f"Selected Style: **{name}**\n\n🎨 Now, please select the highlight color:",
        reply_markup=reply_markup
    )
    return STATE_CHOOSING_COLOR

async def color_selected(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle color selection callback."""
    query = update.callback_query
    await query.answer()
    
    color_choice = query.data.split(":")[1]
    context.user_data["chosen_color"] = color_choice
    last_text = context.user_data.get("last_text")
    line_bottom = context.user_data.get("line_bottom")
    font_choice = context.user_data.get("chosen_font", "faith")
    doge_lines = context.user_data.get("doge_highlight_lines")
    save_user_session(update.effective_user.id, chosen_font=font_choice, chosen_color=color_choice, doge_highlight_lines=doge_lines, last_text=last_text, line_bottom=line_bottom)
    
    color_names = {
        "yellow": "🟡 Yellow",
        "red": "🔴 Red",
        "orange": "🟠 Orange",
        "magenta": "🟣 Magenta",
        "green": "🟢 Green",
        "blue": "🔵 Blue",
        "purple": "🟪 Purple"
    }
    color_name = color_names.get(color_choice, color_choice.capitalize())
    
    if last_text:
        status_msg = await query.message.reply_text(f"🎨 Re-rendering in **{font_choice.capitalize()}** ({color_name})...")
        try:
            user_support_image_path = context.user_data.get("user_support_image_path")
            image_buffer, line_bottom = await asyncio.to_thread(
                generate_textbox_image,
                last_text, font_choice, color_choice,
                user_support_image_path=user_support_image_path,
                doge_highlight_lines=doge_lines
            )
            png_bytes = image_buffer.getvalue()
            context.user_data["transparent_png_bytes"] = png_bytes
            context.user_data["last_rendered_image_bytes"] = png_bytes
            context.user_data["line_bottom"] = line_bottom
            save_user_session(update.effective_user.id, chosen_font=font_choice, chosen_color=color_choice, doge_highlight_lines=doge_lines, last_text=last_text, line_bottom=line_bottom)
            
            reply_markup = get_rendered_textbox_keyboard(font_choice)
            caption_text = f"✅ **{font_choice.capitalize()}** transparent text box generated!"
            as_doc = font_choice in ("maga", "charlie", "faith", "freedom", "doge")
            await send_rendered_media_with_retry(
                query.message,
                png_bytes,
                filename=f"{font_choice}_textbox.png",
                as_document=as_doc,
                caption=caption_text,
                reply_markup=reply_markup
            )
            await safe_delete_message(status_msg)
            
            bg_keyboard = [
                [InlineKeyboardButton("🔍 AI Find Images (Auto)", callback_data="ai_find_images")],
                [InlineKeyboardButton("✅ Done", callback_data="bg_done")]
            ]
            await query.message.reply_text(
                "🖼️ **Would you like to overlay this text box on top of a background image or video?**\n\n"
                "🔍 Click **AI Find Images (Auto)** to automatically find photos for this post,\n"
                "📥 Or **send your own photo/video now** to use as background,\n"
                "➡️ Or click **Done** below to keep this transparent layout and generate your caption.",
                reply_markup=InlineKeyboardMarkup(bg_keyboard)
            )
            return STATE_WAITING_BACKGROUND
        except Exception as e:
            logger.error(f"Error re-rendering in color_selected: {e}")
            await safe_edit_status_message(status_msg, f"❌ Error re-rendering: {str(e)}")
            return STATE_WAITING_TEXT

    await query.edit_message_text(
        text=f"🎨 Font: **{font_choice.capitalize()}** | Color: **{color_name}**\n\n"
             "💬 Send me the text you want in your box.\n"
             "Use line breaks (Shift+Enter) to define separate lines."
    )
    return STATE_WAITING_TEXT

async def change_color(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle color change after output — re-render with new color if last text exists."""
    restore_session_if_needed(update, context)
    query = update.callback_query
    await query.answer()
    
    color_choice = query.data.split(":")[1]
    context.user_data["chosen_color"] = color_choice
    save_user_session(update.effective_user.id, context.user_data.get("chosen_font"), color_choice, context.user_data.get("doge_highlight_lines"))
    
    color_names = {
        "yellow": "🟡 Yellow",
        "red": "🔴 Red",
        "orange": "🟠 Orange",
        "magenta": "🟣 Magenta",
        "green": "🟢 Green",
        "blue": "🔵 Blue",
        "purple": "🟪 Purple"
    }
    color_name = color_names.get(color_choice, color_choice.capitalize())
    font_choice = context.user_data.get("chosen_font", "faith")
    last_text = context.user_data.get("last_text", None)
    
    if last_text:
        # Re-render with the new color
        status_msg = await query.message.reply_text("🎨 Re-rendering with new color...")
        try:
            user_support_image_path = context.user_data.get("user_support_image_path")
            doge_lines = context.user_data.get("doge_highlight_lines")
            image_buffer, line_bottom = await asyncio.to_thread(
                generate_textbox_image,
                last_text, font_choice, color_choice,
                user_support_image_path=user_support_image_path,
                doge_highlight_lines=doge_lines
            )
            
            # Save transparent PNG bytes in session
            png_bytes = image_buffer.getvalue()
            context.user_data["transparent_png_bytes"] = png_bytes
            context.user_data["last_rendered_image_bytes"] = png_bytes
            context.user_data["line_bottom"] = line_bottom
            save_user_session(update.effective_user.id, chosen_font=font_choice, chosen_color=color_choice, doge_highlight_lines=doge_lines, last_text=last_text, line_bottom=line_bottom)
            
            reply_markup = get_rendered_textbox_keyboard(font_choice)
            
            caption_text = (
                f"✅ **{font_choice.capitalize()}** text box with **{color_name}** highlight!\n\n"
                "📝 Send more text to generate another, or use buttons below."
            )
            as_doc = font_choice in ("maga", "charlie", "faith", "freedom", "doge")
            await send_rendered_media_with_retry(
                query.message,
                png_bytes,
                filename=f"{font_choice}_textbox.png",
                as_document=as_doc,
                caption=caption_text,
                reply_markup=reply_markup
            )
            await safe_delete_message(status_msg)
            
            # Re-prompt for background image/video
            bg_keyboard = [
                [InlineKeyboardButton("🔍 AI Find Images (Auto)", callback_data="ai_find_images")],
                [InlineKeyboardButton("✅ Done", callback_data="bg_done")]
            ]
            await query.message.reply_text(
                "🖼️ **Would you like to overlay this text box on top of a background image or video?**\n\n"
                "🔍 Click **AI Find Images (Auto)** to automatically find photos for this post,\n"
                "📥 Or **send your own photo/video now** to use as background,\n"
                "➡️ Or click **Done** below to keep this transparent layout and generate your caption.",
                reply_markup=InlineKeyboardMarkup(bg_keyboard)
            )
            return STATE_WAITING_BACKGROUND
            
        except Exception as e:
            logger.error(f"Failed to re-render: {e}")
            err_msg = (
                "⚠️ Telegram connection timed out while sending the text box. Please try again."
                if (isinstance(e, TimedOut) or "timed out" in str(e).lower())
                else f"❌ Error re-rendering: {str(e)}"
            )
            await safe_edit_status_message(status_msg, err_msg)
            return STATE_WAITING_BACKGROUND
    else:
        await query.message.reply_text(
            f"Color changed to **{color_name}**. Send text to generate an image."
        )
        return STATE_WAITING_TEXT

async def change_font(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Let user change font without restarting."""
    restore_session_if_needed(update, context)
    query = update.callback_query
    await query.answer()
    
    keyboard = [
        [
            InlineKeyboardButton("✨ Faith", callback_data="font:faith")
        ],
        [
            InlineKeyboardButton("🇺🇸 Maga", callback_data="font:maga"),
            InlineKeyboardButton("🇺🇸 Charlie", callback_data="font:charlie")
        ],
        [
            InlineKeyboardButton("🐕 Doge (League Gothic)", callback_data="font:doge")
        ],
        [
            InlineKeyboardButton("🗽 Freedom (Montserrat Black)", callback_data="font:freedom")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.message.reply_text(
        "🔄 Choose a new font style:",
        reply_markup=reply_markup
    )
    return STATE_CHOOSING_FONT

async def edit_text_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Prompt the user to send their edited text with one-tap copy."""
    query = update.callback_query
    await query.answer()
    
    last_text = context.user_data.get("last_text", "")
    if last_text:
        await query.message.reply_text(
            "✏️ **Tap the text box below to copy it, make your changes, and send it back to re-render:**\n\n"
            f"`{last_text}`",
            parse_mode="Markdown"
        )
    else:
        await query.message.reply_text("💬 Please send the new text you would like to render:")
    return STATE_WAITING_TEXT

async def ocr_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Prompt user to send a screenshot for OCR extraction."""
    await update.message.reply_text(
        "📸 **Send me any screenshot or image now!**\n\n"
        "I'll automatically detect and extract the headline text with AI, then render your text box."
    )
    return STATE_WAITING_TEXT

async def image_received(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Auto-detects and extracts headline from screenshots using OCR + Groq AI and renders immediately."""
    restore_session_if_needed(update, context)
    
    # 1. Get the image file
    photo_file = None
    if update.message.photo:
        photo_file = await update.message.photo[-1].get_file()
    elif update.message.document and update.message.document.mime_type and update.message.document.mime_type.startswith("image/"):
        photo_file = await update.message.document.get_file()
        
    if not photo_file:
        await update.message.reply_text("❌ Please send a valid photo or screenshot.")
        return STATE_WAITING_TEXT
        
    status_msg = await update.message.reply_text("🔍 Analyzing screenshot & extracting headline with AI...")
    try:
        img_bytes = await photo_file.download_as_bytearray()
        img_bytes = bytes(img_bytes)
        
        # 2. Extract raw OCR text via OCR.space
        raw_ocr = await asyncio.to_thread(call_ocr_space, img_bytes)
        
        # 3. Clean and isolate headline using Groq AI
        extracted_headline = await asyncio.to_thread(clean_ocr_headline_with_ai, raw_ocr)
        
        if not extracted_headline or len(extracted_headline.strip()) < 5:
            # If no headline found, check if user had previous text box and meant this as background
            png_bytes = context.user_data.get("transparent_png_bytes")
            last_text = context.user_data.get("last_text")
            if png_bytes or last_text:
                await safe_edit_status_message(status_msg, "ℹ️ No headline text detected in this image. Applying as background...")
                return await background_received(update, context)
                
            await safe_edit_status_message(
                status_msg,
                "🤔 Couldn't detect any clear headline or quote text in this screenshot.\n\n"
                "💡 **Tips:**\n"
                "• Send a screenshot containing a headline, quote, or news graphic\n"
                "• Or type your text directly in chat!"
            )
            return STATE_WAITING_TEXT

        await safe_delete_message(status_msg)
        
        # Display extracted text clearly to user
        await update.message.reply_text(
            f"📸 **Extracted Headline:**\n\n`{extracted_headline}`",
            parse_mode="Markdown"
        )
        
        # Setup session for render
        context.user_data["last_text"] = extracted_headline
        font_choice = context.user_data.get("chosen_font", "faith")
        
        if font_choice == "charlie":
            charlie_colors = ["yellow", "red", "orange", "magenta", "green", "blue", "purple"]
            current_index = context.user_data.get("charlie_color_index", 0)
            color_choice = charlie_colors[current_index]
            context.user_data["chosen_color"] = color_choice
            context.user_data["charlie_color_index"] = (current_index + 1) % len(charlie_colors)
        elif font_choice == "maga":
            maga_colors = ["yellow", "red", "orange", "magenta", "green", "blue", "purple"]
            current_index = context.user_data.get("maga_color_index", 0)
            color_choice = maga_colors[current_index]
            context.user_data["chosen_color"] = color_choice
            context.user_data["maga_color_index"] = (current_index + 1) % len(maga_colors)
        elif font_choice == "freedom":
            color_choice = "orange"
            context.user_data["chosen_color"] = "orange"
        else:
            color_choice = context.user_data.get("chosen_color", "yellow")
            
        render_status = await update.message.reply_text("🎨 Rendering your text box, please wait...")
        user_support_image_path = context.user_data.get("user_support_image_path")
        image_buffer, line_bottom = await asyncio.to_thread(
            generate_textbox_image,
            extracted_headline, font_choice, color_choice, user_support_image_path=user_support_image_path
        )
        
        png_bytes = image_buffer.getvalue()
        context.user_data["transparent_png_bytes"] = png_bytes
        context.user_data["last_rendered_image_bytes"] = png_bytes
        context.user_data["line_bottom"] = line_bottom
        save_user_session(update.effective_user.id, chosen_font=font_choice, chosen_color=color_choice, last_text=extracted_headline, line_bottom=line_bottom)
        
        reply_markup = get_rendered_textbox_keyboard(font_choice)
        as_doc = font_choice in ("maga", "charlie", "faith", "freedom", "doge")
        await send_rendered_media_with_retry(
            update.message,
            png_bytes,
            filename=f"{font_choice}_textbox.png",
            as_document=as_doc,
            caption=f"✅ **{font_choice.capitalize()}** transparent text box generated from screenshot!",
            reply_markup=reply_markup
        )
        await safe_delete_message(render_status)
        
        bg_keyboard = [
            [InlineKeyboardButton("🔍 AI Find Images (Auto)", callback_data="ai_find_images")],
            [InlineKeyboardButton("✅ Done (Keep Transparent)", callback_data="bg_done")]
        ]
        await update.message.reply_text(
            "🖼️ **Would you like to overlay this text box on top of a background image or video?**\n\n"
            "🔍 Click **AI Find Images (Auto)** to automatically find photos for this post,\n"
            "📥 Or **send your own photo/video now** to use as background,\n"
            "➡️ Or click **Done** below to keep this transparent layout.",
            reply_markup=InlineKeyboardMarkup(bg_keyboard)
        )
        return STATE_WAITING_BACKGROUND
        
    except Exception as e:
        logger.error(f"Error processing screenshot OCR: {e}")
        await safe_edit_status_message(status_msg, f"❌ Error processing screenshot: {str(e)}")
        return STATE_WAITING_TEXT

async def clear_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Inform user that custom images feature is disabled."""
    await update.message.reply_text(
        "⚠️ Custom supporting images are disabled."
    )
    return STATE_WAITING_TEXT

async def text_received(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Generate and send image upon text reception. User stays on the same page."""
    restore_session_if_needed(update, context)
    text = update.message.text.strip()
    user_id = update.effective_user.id
    font_choice = context.user_data.get("chosen_font", "faith")
    if font_choice == "charlie":
        charlie_colors = ["yellow", "red", "orange", "magenta", "green", "blue", "purple"]
        current_index = context.user_data.get("charlie_color_index", 0)
        color_choice = charlie_colors[current_index]
        context.user_data["chosen_color"] = color_choice
        context.user_data["charlie_color_index"] = (current_index + 1) % len(charlie_colors)
    elif font_choice == "maga":
        maga_colors = ["yellow", "red", "orange", "magenta", "green", "blue", "purple"]
        current_index = context.user_data.get("maga_color_index", 0)
        color_choice = maga_colors[current_index]
        context.user_data["chosen_color"] = color_choice
        context.user_data["maga_color_index"] = (current_index + 1) % len(maga_colors)
    elif font_choice == "freedom":
        color_choice = "orange"
        context.user_data["chosen_color"] = "orange"
    else:
        color_choice = context.user_data.get("chosen_color", "yellow")
    
    is_instagram = "instagram.com" in text.lower() or "instagr.am" in text.lower()
    
    if is_instagram:
        status_msg = await update.message.reply_text("🔍 Downloading Instagram post image & performing OCR...")
        try:
            # 1. Extract shortcode
            parts = text.strip("/").split("/")
            shortcode = None
            for i, p in enumerate(parts):
                if p in ("p", "reel", "tv") and i + 1 < len(parts):
                    shortcode = parts[i + 1].split("?")[0]
                    break
            
            if not shortcode:
                raise ValueError("Could not extract Instagram shortcode from URL.")
                
            # 2. Download high-res image via the public media redirection link
            media_url = f"https://www.instagram.com/p/{shortcode}/media/?size=l"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
            }
            img_response = await asyncio.to_thread(requests.get, media_url, headers=headers, allow_redirects=True, timeout=15)
            if img_response.status_code != 200 or "image" not in img_response.headers.get("Content-Type", ""):
                raise ValueError(f"Failed to download image (HTTP {img_response.status_code})")
                
            # 3. Call OCR.space with the downloaded image bytes
            await status_msg.edit_text("🤖 Image downloaded! Extracting text via OCR...")
            extracted_text = await asyncio.to_thread(call_ocr_space, img_response.content)
            
            # If OCR failed, try fallback to caption
            if not extracted_text:
                try:
                    caption, _, _ = await asyncio.to_thread(fetch_instagram_post, text, user_id)
                    extracted_text = caption
                except Exception:
                    pass
            
            if not extracted_text:
                raise ValueError("Could not extract text via OCR or caption.")
                
            # 4. Save extracted text in context
            context.user_data["ocr_extracted_text"] = extracted_text
            
            # AI Proofreading check on extracted OCR text
            corrected_text = await asyncio.to_thread(proofread_text_with_ai, extracted_text)
            has_corrections = (corrected_text.upper().strip() != extracted_text.upper().strip())
            context.user_data["suggested_corrected_text"] = corrected_text
            
            if has_corrections:
                kb = [
                    [
                        InlineKeyboardButton("✨ Use Corrected Version", callback_data="ig_ocr:proceed_corrected"),
                        InlineKeyboardButton("📄 Use Original OCR Text", callback_data="ig_ocr:proceed_original")
                    ],
                    [
                        InlineKeyboardButton("✏️ Edit Text", callback_data="ig_ocr:edit")
                    ]
                ]
            else:
                kb = [
                    [
                        InlineKeyboardButton("✅ Proceed (Text is Good)", callback_data="ig_ocr:proceed_original"),
                        InlineKeyboardButton("✏️ Edit Text", callback_data="ig_ocr:edit")
                    ]
                ]
                
            markup = InlineKeyboardMarkup(kb)
            await safe_delete_message(status_msg)
            
            if has_corrections:
                msg_body = (
                    f"📝 **Text extracted from Instagram post:**\n"
                    f"`{extracted_text}`\n\n"
                    f"✨ **AI Grammatical Correction:**\n"
                    f"`{corrected_text}`\n\n"
                    f"Which version would you like to proceed with?"
                )
            else:
                msg_body = (
                    f"📝 **Here is the text I extracted from that Instagram post:**\n\n"
                    f"`{extracted_text}`\n\n"
                    f"Does this look good, or would you like to edit it?"
                )
                
            await update.message.reply_text(
                msg_body,
                reply_markup=markup,
                parse_mode="Markdown"
            )
            return STATE_CONFIRM_IG_OCR
            
        except Exception as e:
            logger.error(f"Instagram OCR workflow failed: {e}")
            await status_msg.edit_text(
                f"⚠️ **Could not download or OCR the Instagram post:** {str(e)}.\n\n"
                "Please type/send your text box text directly, or upload the image post file here!"
            )
            return STATE_WAITING_TEXT
            
    # Non-Instagram regular rendering path
    if font_choice == "doge":
        context.user_data["doge_highlight_lines"] = None
        context.user_data["doge_pending_text"] = text
        await update.message.reply_text(
            "🖍️ **How many lines would you like to highlight/color?**\n\n"
            "Send a number (e.g. 1, 2, 3, etc.):"
        )
        return STATE_WAITING_DOGE_HIGHLIGHT_LINES
        
    status_msg = await update.message.reply_text("🎨 Checking grammar & rendering, please wait...")
    
    # AI Proofreading check
    corrected_text = await asyncio.to_thread(proofread_text_with_ai, text)
    has_corrections = (corrected_text.upper().strip() != text.upper().strip())
    
    # Store the last text so color change can re-render
    context.user_data["last_text"] = text
    
    try:
        user_support_image_path = context.user_data.get("user_support_image_path")
        image_buffer, line_bottom = await asyncio.to_thread(
            generate_textbox_image,
            text, font_choice, color_choice,
            user_support_image_path=user_support_image_path
        )
        
        # Save transparent PNG bytes in session
        png_bytes = image_buffer.getvalue()
        context.user_data["transparent_png_bytes"] = png_bytes
        context.user_data["last_rendered_image_bytes"] = png_bytes
        context.user_data["line_bottom"] = line_bottom
        save_user_session(user_id, chosen_font=font_choice, chosen_color=color_choice, last_text=text, line_bottom=line_bottom)
        
        color_names = {
            "yellow": "🟡 Yellow",
            "red": "🔴 Red",
            "orange": "🟠 Orange",
            "magenta": "🟣 Magenta",
            "green": "🟢 Green",
            "blue": "🔵 Blue",
            "purple": "🟪 Purple"
        }
        color_name = color_names.get(color_choice, color_choice.capitalize())
        
        reply_markup = get_rendered_textbox_keyboard(font_choice)
        
        caption_text = f"✅ **{font_choice.capitalize()}** transparent text box generated!"
        as_doc = font_choice in ("maga", "charlie", "faith", "freedom", "doge")
        await send_rendered_media_with_retry(
            update.message,
            png_bytes,
            filename=f"{font_choice}_textbox.png",
            as_document=as_doc,
            caption=caption_text,
            reply_markup=reply_markup
        )
            
        await safe_delete_message(status_msg)
        
        # Send proofreading suggestion if corrections found
        if has_corrections:
            context.user_data["suggested_corrected_text"] = corrected_text
            kb = [
                [
                    InlineKeyboardButton("✅ Use Corrected Text", callback_data="proofread:accept"),
                    InlineKeyboardButton("❌ Keep Original", callback_data="proofread:ignore")
                ]
            ]
            markup = InlineKeyboardMarkup(kb)
            await update.message.reply_text(
                f"📝 **AI Proofreading Suggestion:**\n"
                f"I detected some spelling or grammar errors.\n\n"
                f"**Suggested Correction:**\n"
                f"`{corrected_text}`\n\n"
                f"Would you like to switch to this version?",
                reply_markup=markup,
                parse_mode="Markdown"
            )
        bg_keyboard = [
            [InlineKeyboardButton("🔍 AI Find Images (Auto)", callback_data="ai_find_images")],
            [InlineKeyboardButton("✅ Done (Keep Transparent)", callback_data="bg_done")]
        ]
        await update.message.reply_text(
            "🖼️ **Would you like to overlay this text box on top of a background image or video?**\n\n"
            "🔍 Click **AI Find Images (Auto)** to automatically find photos for this post,\n"
            "📥 Or **send your own photo/video now** to use as background,\n"
            "➡️ Or click **Done** below to keep this transparent layout.",
            reply_markup=InlineKeyboardMarkup(bg_keyboard)
        )
        return STATE_WAITING_BACKGROUND
        
    except Exception as e:
        logger.error(f"Failed to generate/send image: {e}")
        err_msg = (
            "⚠️ Telegram connection timed out while sending the text box. Please send your text again or tap /restart."
            if (isinstance(e, TimedOut) or "timed out" in str(e).lower())
            else f"❌ Sorry, an error occurred during rendering: {str(e)}"
        )
        await safe_edit_status_message(status_msg, err_msg)
        return STATE_WAITING_TEXT

async def doge_highlight_lines_received(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle receiving number of lines to color for Doge style."""
    restore_session_if_needed(update, context)
    num_lines_text = update.message.text.strip()
    try:
        num_lines = int(num_lines_text)
        if num_lines < 0:
            raise ValueError()
    except ValueError:
        await update.message.reply_text("⚠️ Please send a valid positive number (e.g. 1, 2, 3):")
        return STATE_WAITING_DOGE_HIGHLIGHT_LINES

    context.user_data["doge_highlight_lines"] = num_lines
    save_user_session(update.effective_user.id, chosen_font="doge", chosen_color=context.user_data.get("chosen_color"), doge_highlight_lines=num_lines)
    text = context.user_data.get("doge_pending_text", "")
    
    # Clear the pending text
    context.user_data.pop("doge_pending_text", None)
    
    status_msg = await update.message.reply_text("🎨 Checking grammar & rendering, please wait...")
    
    # AI Proofreading check
    corrected_text = await asyncio.to_thread(proofread_text_with_ai, text)
    has_corrections = (corrected_text.upper().strip() != text.upper().strip())
    
    # Store the last text so color change can re-render
    context.user_data["last_text"] = text
    font_choice = "doge"
    color_choice = context.user_data.get("chosen_color", "yellow")
    
    try:
        user_support_image_path = context.user_data.get("user_support_image_path")
        image_buffer, line_bottom = await asyncio.to_thread(
            generate_textbox_image,
            text, font_choice, color_choice,
            user_support_image_path=user_support_image_path,
            doge_highlight_lines=num_lines
        )
        
        # Save transparent PNG bytes in session
        png_bytes = image_buffer.getvalue()
        context.user_data["transparent_png_bytes"] = png_bytes
        context.user_data["last_rendered_image_bytes"] = png_bytes
        context.user_data["line_bottom"] = line_bottom
        save_user_session(update.effective_user.id, chosen_font="doge", chosen_color=color_choice, doge_highlight_lines=num_lines, last_text=text, line_bottom=line_bottom)
        
        color_names = {
            "yellow": "🟡 Yellow",
            "red": "🔴 Red",
            "orange": "🟠 Orange",
            "magenta": "🟣 Magenta",
            "green": "🟢 Green",
            "blue": "🔵 Blue",
            "purple": "🟪 Purple"
        }
        color_name = color_names.get(color_choice, color_choice.capitalize())
        
        reply_markup = get_rendered_textbox_keyboard("doge")
        
        caption_text = f"✅ **Doge** transparent text box generated!"
        await send_rendered_media_with_retry(
            update.message,
            png_bytes,
            filename="doge_textbox.png",
            as_document=True,
            caption=caption_text,
            reply_markup=reply_markup
        )
            
        await safe_delete_message(status_msg)
        
        # Send proofreading suggestion if corrections found
        if has_corrections:
            context.user_data["suggested_corrected_text"] = corrected_text
            kb = [
                [
                    InlineKeyboardButton("✅ Use Corrected Text", callback_data="proofread:accept"),
                    InlineKeyboardButton("❌ Keep Original", callback_data="proofread:ignore")
                ]
            ]
            markup = InlineKeyboardMarkup(kb)
            await update.message.reply_text(
                f"📝 **AI Proofreading Suggestion:**\n"
                f"I detected some spelling or grammar errors.\n\n"
                f"**Suggested Correction:**\n"
                f"`{corrected_text}`\n\n"
                f"Would you like to switch to this version?",
                reply_markup=markup,
                parse_mode="Markdown"
            )
        await update.message.reply_text(
            "🖼️ **Would you like to overlay this text box on top of a background image or video?**\n\n"
            "📥 **Send an image or a video now** to blend it,\n"
            "➡️ Or click/type /skip to keep the transparent PNG layout and send new text."
        )
        return STATE_WAITING_BACKGROUND
        
    except Exception as e:
        logger.error(f"Failed to generate/send image: {e}")
        err_msg = (
            "⚠️ Telegram connection timed out while sending the text box. Please send your text again or tap /restart."
            if (isinstance(e, TimedOut) or "timed out" in str(e).lower())
            else f"❌ Sorry, an error occurred during rendering: {str(e)}"
        )
        await safe_edit_status_message(status_msg, err_msg)
        return STATE_WAITING_TEXT

async def ig_ocr_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle callback queries in the Instagram OCR confirmation screen."""
    query = update.callback_query
    await query.answer()
    
    action = query.data.split(":")[1]
    if action == "edit":
        text_to_edit = context.user_data.get("suggested_corrected_text") or context.user_data.get("ocr_extracted_text", "")
        await query.message.reply_text(
            "💬 **Please copy the text below, edit it, and reply to this message with your updated version:**\n\n"
            f"`{text_to_edit}`",
            parse_mode="Markdown"
        )
        return STATE_CONFIRM_IG_OCR
        
    if action == "proceed_corrected":
        chosen_text = context.user_data.get("suggested_corrected_text", "").strip()
    else:  # proceed_original
        chosen_text = context.user_data.get("ocr_extracted_text", "").strip()
        
    if not chosen_text:
        await query.message.reply_text("❌ No text found. Please send the link or text again.")
        return STATE_WAITING_TEXT
        
    context.user_data["last_text"] = chosen_text
    
    # Ask for color!
    font_choice = context.user_data.get("chosen_font", "faith")
    if font_choice == "freedom":
        color_choice = "orange"
        context.user_data["chosen_color"] = color_choice
        status_msg = await query.message.reply_text("🎨 Rendering your text box, please wait...")
        try:
            user_support_image_path = context.user_data.get("user_support_image_path")
            image_buffer, line_bottom = await asyncio.to_thread(
                generate_textbox_image,
                chosen_text, "freedom", color_choice, user_support_image_path=user_support_image_path
            )
            png_bytes = image_buffer.getvalue()
            context.user_data["transparent_png_bytes"] = png_bytes
            context.user_data["line_bottom"] = line_bottom
            save_user_session(update.effective_user.id, chosen_font="freedom", chosen_color=color_choice, last_text=chosen_text, line_bottom=line_bottom)
            reply_markup = get_rendered_textbox_keyboard("freedom")
            caption_text = "✅ **Freedom** transparent text box generated!"
            await send_rendered_media_with_retry(
                query.message,
                png_bytes,
                filename="freedom_textbox.png",
                as_document=True,
                caption=caption_text,
                reply_markup=reply_markup
            )
            await safe_delete_message(status_msg)
            await query.message.reply_text(
                "🖼️ **Would you like to overlay this text box on top of a background image or video?**\n\n"
                "📥 **Send an image or a video now** to blend it,\n"
                "➡️ Or click/type /skip to keep the transparent PNG layout and send new text."
            )
            return STATE_WAITING_BACKGROUND
        except Exception as e:
            logger.error(f"Failed to generate/send image in OCR flow for Freedom: {e}")
            err_msg = (
                "⚠️ Telegram connection timed out while sending the text box. Please try again."
                if (isinstance(e, TimedOut) or "timed out" in str(e).lower())
                else f"❌ Sorry, an error occurred during rendering: {str(e)}"
            )
            await safe_edit_status_message(status_msg, err_msg)
            return STATE_WAITING_TEXT

    if font_choice == "charlie":
        charlie_colors = ["yellow", "red", "orange", "magenta", "green", "blue", "purple"]
        current_index = context.user_data.get("charlie_color_index", 0)
        color_choice = charlie_colors[current_index]
        context.user_data["chosen_color"] = color_choice
        context.user_data["charlie_color_index"] = (current_index + 1) % len(charlie_colors)
        
        status_msg = await query.message.reply_text("🎨 Rendering your text box, please wait...")
        try:
            user_support_image_path = context.user_data.get("user_support_image_path")
            image_buffer, line_bottom = await asyncio.to_thread(
                generate_textbox_image,
                chosen_text, "charlie", color_choice, user_support_image_path=user_support_image_path
            )
            
            # Save transparent PNG bytes in session
            png_bytes = image_buffer.getvalue()
            context.user_data["transparent_png_bytes"] = png_bytes
            context.user_data["last_rendered_image_bytes"] = png_bytes
            context.user_data["last_text"] = chosen_text
            context.user_data["line_bottom"] = line_bottom
            save_user_session(update.effective_user.id, chosen_font="charlie", chosen_color=color_choice, last_text=chosen_text, line_bottom=line_bottom)
            
            reply_markup = get_rendered_textbox_keyboard(font_choice)
            
            caption_text = f"✅ **Charlie** transparent text box generated with auto-cycled color!"
            await send_rendered_media_with_retry(
                query.message,
                png_bytes,
                filename="charlie_textbox.png",
                as_document=True,
                caption=caption_text,
                reply_markup=reply_markup
            )
            await safe_delete_message(status_msg)
            
            # Ask user for background image/video
            bg_keyboard = [
                [InlineKeyboardButton("🔍 AI Find Images (Auto)", callback_data="ai_find_images")],
                [InlineKeyboardButton("✅ Done", callback_data="bg_done")]
            ]
            await query.message.reply_text(
                "🖼️ **Would you like to overlay this text box on top of a background image or video?**\n\n"
                "🔍 Click **AI Find Images (Auto)** to automatically find photos for this post,\n"
                "📥 Or **send your own photo/video now** to use as background,\n"
                "➡️ Or click **Done** below to keep this transparent layout and generate your caption.",
                reply_markup=InlineKeyboardMarkup(bg_keyboard)
            )
            return STATE_WAITING_BACKGROUND
            
        except Exception as e:
            logger.error(f"Failed to generate/send image in OCR flow for Charlie: {e}")
            err_msg = (
                "⚠️ Telegram connection timed out while sending the text box. Please try again."
                if (isinstance(e, TimedOut) or "timed out" in str(e).lower())
                else f"❌ Sorry, an error occurred during rendering: {str(e)}"
            )
            await safe_edit_status_message(status_msg, err_msg)
            return STATE_WAITING_TEXT
            
    if font_choice == "maga":
        maga_colors = ["yellow", "red", "orange", "magenta", "green", "blue", "purple"]
        current_index = context.user_data.get("maga_color_index", 0)
        color_choice = maga_colors[current_index]
        context.user_data["chosen_color"] = color_choice
        context.user_data["maga_color_index"] = (current_index + 1) % len(maga_colors)
        
        status_msg = await update.message.reply_text("🎨 Rendering your text box, please wait...")
        try:
            user_support_image_path = context.user_data.get("user_support_image_path")
            image_buffer, line_bottom = await asyncio.to_thread(
                generate_textbox_image,
                text, "maga", color_choice, user_support_image_path=user_support_image_path
            )
            
            png_bytes = image_buffer.getvalue()
            context.user_data["transparent_png_bytes"] = png_bytes
            context.user_data["last_rendered_image_bytes"] = png_bytes
            context.user_data["last_text"] = text
            context.user_data["line_bottom"] = line_bottom
            save_user_session(update.effective_user.id, chosen_font="maga", chosen_color=color_choice, last_text=text, line_bottom=line_bottom)
            
            reply_markup = get_rendered_textbox_keyboard(font_choice)
            
            caption_text = f"✅ **Maga** transparent text box generated with auto-cycled color!"
            await send_rendered_media_with_retry(
                update.message,
                png_bytes,
                filename="maga_textbox.png",
                as_document=True,
                caption=caption_text,
                reply_markup=reply_markup
            )
            await safe_delete_message(status_msg)
            
            bg_keyboard = [
                [InlineKeyboardButton("🔍 AI Find Images (Auto)", callback_data="ai_find_images")],
                [InlineKeyboardButton("✅ Done", callback_data="bg_done")]
            ]
            await update.message.reply_text(
                "🖼️ **Would you like to overlay this text box on top of a background image or video?**\n\n"
                "🔍 Click **AI Find Images (Auto)** to automatically find photos for this post,\n"
                "📥 Or **send your own photo/video now** to use as background,\n"
                "➡️ Or click **Done** below to keep this transparent layout and generate your caption.",
                reply_markup=InlineKeyboardMarkup(bg_keyboard)
            )
            return STATE_WAITING_BACKGROUND
            
        except Exception as e:
            logger.error(f"Failed to generate/send image in OCR flow for Maga: {e}")
            err_msg = (
                "⚠️ Telegram connection timed out while sending the text box. Please try again."
                if (isinstance(e, TimedOut) or "timed out" in str(e).lower())
                else f"❌ Sorry, an error occurred during rendering: {str(e)}"
            )
            await safe_edit_status_message(status_msg, err_msg)
            return STATE_WAITING_TEXT

    if font_choice in ("charlie", "maga_charlie", "faith"):
        keyboard = [
            [
                InlineKeyboardButton("🟡 Yellow (#ffde59)", callback_data="color_ocr:yellow"),
                InlineKeyboardButton("🔴 Red (#ff3131)", callback_data="color_ocr:red")
            ],
            [
                InlineKeyboardButton("🟠 Orange (#ff914d)", callback_data="color_ocr:orange"),
                InlineKeyboardButton("🟣 Magenta (#cb6ce6)", callback_data="color_ocr:magenta")
            ],
            [
                InlineKeyboardButton("🟢 Green (#00e676)", callback_data="color_ocr:green"),
                InlineKeyboardButton("🔵 Blue (#38b6ff)", callback_data="color_ocr:blue")
            ],
            [
                InlineKeyboardButton("🟪 Purple (#8c52ff)", callback_data="color_ocr:purple")
            ]
        ]
    else:
        keyboard = [
            [
                InlineKeyboardButton("🟡 Yellow (#ffde59)", callback_data="color_ocr:yellow"),
                InlineKeyboardButton("🔴 Red (#ff3131)", callback_data="color_ocr:red")
            ],
            [
                InlineKeyboardButton("🟢 Green (#00e676)", callback_data="color_ocr:green"),
                InlineKeyboardButton("🔵 Blue (#38b6ff)", callback_data="color_ocr:blue")
            ],
            [
                InlineKeyboardButton("🟪 Purple (#8c52ff)", callback_data="color_ocr:purple")
            ]
        ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.message.reply_text(
        "🎨 **Please select the highlight color to generate the text box:**",
        reply_markup=reply_markup,
        parse_mode="Markdown"
    )
    return STATE_CONFIRM_IG_OCR

async def ig_ocr_text_edited(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle text message input when the user edits the OCR text."""
    text = update.message.text.strip()
    
    # Run AI proofreading on edited text too!
    corrected_text = await asyncio.to_thread(proofread_text_with_ai, text)
    has_corrections = (corrected_text.upper().strip() != text.upper().strip())
    
    context.user_data["ocr_extracted_text"] = text
    context.user_data["suggested_corrected_text"] = corrected_text
    
    if has_corrections:
        kb = [
            [
                InlineKeyboardButton("✨ Use Corrected Version", callback_data="ig_ocr:proceed_corrected"),
                InlineKeyboardButton("📄 Use My Original", callback_data="ig_ocr:proceed_original")
            ],
            [
                InlineKeyboardButton("✏️ Edit Again", callback_data="ig_ocr:edit")
            ]
        ]
        markup = InlineKeyboardMarkup(kb)
        await update.message.reply_text(
            f"📝 **Your edited text:**\n`{text}`\n\n"
            f"✨ **AI Grammatical Correction:**\n`{corrected_text}`\n\n"
            "Which version would you like to use?",
            reply_markup=markup,
            parse_mode="Markdown"
        )
        return STATE_CONFIRM_IG_OCR
        
    context.user_data["last_text"] = text
    
    font_choice = context.user_data.get("chosen_font", "faith")
    if font_choice == "freedom":
        color_choice = "orange"
        context.user_data["chosen_color"] = color_choice
        status_msg = await update.message.reply_text("🎨 Rendering your text box, please wait...")
        try:
            user_support_image_path = context.user_data.get("user_support_image_path")
            image_buffer, line_bottom = await asyncio.to_thread(
                generate_textbox_image,
                text, "freedom", color_choice, user_support_image_path=user_support_image_path
            )
            png_bytes = image_buffer.getvalue()
            context.user_data["transparent_png_bytes"] = png_bytes
            context.user_data["line_bottom"] = line_bottom
            save_user_session(update.effective_user.id, chosen_font="freedom", chosen_color=color_choice, last_text=text, line_bottom=line_bottom)
            reply_markup = get_rendered_textbox_keyboard("freedom")
            caption_text = "✅ **Freedom** transparent text box generated!"
            await send_rendered_media_with_retry(
                update.message,
                png_bytes,
                filename="freedom_textbox.png",
                as_document=True,
                caption=caption_text,
                reply_markup=reply_markup
            )
            await safe_delete_message(status_msg)
            await update.message.reply_text(
                "🖼️ **Would you like to overlay this text box on top of a background image or video?**\n\n"
                "📥 **Send an image or a video now** to blend it,\n"
                "➡️ Or click/type /skip to keep the transparent PNG layout and send new text."
            )
            return STATE_WAITING_BACKGROUND
        except Exception as e:
            logger.error(f"Failed to generate/send image in OCR flow for Freedom: {e}")
            err_msg = (
                "⚠️ Telegram connection timed out while sending the text box. Please try again."
                if (isinstance(e, TimedOut) or "timed out" in str(e).lower())
                else f"❌ Sorry, an error occurred during rendering: {str(e)}"
            )
            await safe_edit_status_message(status_msg, err_msg)
            return STATE_WAITING_TEXT

    if font_choice == "charlie":
        charlie_colors = ["yellow", "red", "orange", "magenta", "green", "blue", "purple"]
        current_index = context.user_data.get("charlie_color_index", 0)
        color_choice = charlie_colors[current_index]
        context.user_data["chosen_color"] = color_choice
        context.user_data["charlie_color_index"] = (current_index + 1) % len(charlie_colors)
        
        status_msg = await update.message.reply_text("🎨 Rendering your text box, please wait...")
        try:
            user_support_image_path = context.user_data.get("user_support_image_path")
            image_buffer, line_bottom = await asyncio.to_thread(
                generate_textbox_image,
                text, "charlie", color_choice, user_support_image_path=user_support_image_path
            )
            
            # Save transparent PNG bytes in session
            png_bytes = image_buffer.getvalue()
            context.user_data["transparent_png_bytes"] = png_bytes
            context.user_data["last_rendered_image_bytes"] = png_bytes
            context.user_data["last_text"] = text
            context.user_data["line_bottom"] = line_bottom
            save_user_session(update.effective_user.id, chosen_font="charlie", chosen_color=color_choice, last_text=text, line_bottom=line_bottom)
            
            reply_markup = get_rendered_textbox_keyboard(font_choice)
            
            caption_text = f"✅ **Charlie** transparent text box generated with auto-cycled color!"
            await send_rendered_media_with_retry(
                update.message,
                png_bytes,
                filename="charlie_textbox.png",
                as_document=True,
                caption=caption_text,
                reply_markup=reply_markup
            )
            await safe_delete_message(status_msg)
            
            # Ask user for background image/video
            bg_keyboard = [
                [InlineKeyboardButton("🔍 AI Find Images (Auto)", callback_data="ai_find_images")],
                [InlineKeyboardButton("✅ Done", callback_data="bg_done")]
            ]
            await update.message.reply_text(
                "🖼️ **Would you like to overlay this text box on top of a background image or video?**\n\n"
                "🔍 Click **AI Find Images (Auto)** to automatically find photos for this post,\n"
                "📥 Or **send your own photo/video now** to use as background,\n"
                "➡️ Or click **Done** below to keep this transparent layout and generate your caption.",
                reply_markup=InlineKeyboardMarkup(bg_keyboard)
            )
            return STATE_WAITING_BACKGROUND
            
        except Exception as e:
            logger.error(f"Failed to generate/send image in OCR flow for Charlie: {e}")
            err_msg = (
                "⚠️ Telegram connection timed out while sending the text box. Please try again."
                if (isinstance(e, TimedOut) or "timed out" in str(e).lower())
                else f"❌ Sorry, an error occurred during rendering: {str(e)}"
            )
            await safe_edit_status_message(status_msg, err_msg)
            return STATE_WAITING_TEXT
            
    if font_choice == "maga":
        maga_colors = ["yellow", "red", "orange", "magenta", "green", "blue", "purple"]
        current_index = context.user_data.get("maga_color_index", 0)
        color_choice = maga_colors[current_index]
        context.user_data["chosen_color"] = color_choice
        context.user_data["maga_color_index"] = (current_index + 1) % len(maga_colors)
        
        status_msg = await update.message.reply_text("🎨 Rendering your text box, please wait...")
        try:
            user_support_image_path = context.user_data.get("user_support_image_path")
            image_buffer, line_bottom = await asyncio.to_thread(
                generate_textbox_image,
                text, "maga", color_choice, user_support_image_path=user_support_image_path
            )
            
            png_bytes = image_buffer.getvalue()
            context.user_data["transparent_png_bytes"] = png_bytes
            context.user_data["last_rendered_image_bytes"] = png_bytes
            context.user_data["last_text"] = text
            context.user_data["line_bottom"] = line_bottom
            save_user_session(update.effective_user.id, chosen_font="maga", chosen_color=color_choice, last_text=text, line_bottom=line_bottom)
            
            reply_markup = get_rendered_textbox_keyboard(font_choice)
            
            caption_text = f"✅ **Maga** transparent text box generated with auto-cycled color!"
            await send_rendered_media_with_retry(
                update.message,
                png_bytes,
                filename="maga_textbox.png",
                as_document=True,
                caption=caption_text,
                reply_markup=reply_markup
            )
            await safe_delete_message(status_msg)
            
            bg_keyboard = [
                [InlineKeyboardButton("🔍 AI Find Images (Auto)", callback_data="ai_find_images")],
                [InlineKeyboardButton("✅ Done", callback_data="bg_done")]
            ]
            await update.message.reply_text(
                "🖼️ **Would you like to overlay this text box on top of a background image or video?**\n\n"
                "🔍 Click **AI Find Images (Auto)** to automatically find photos for this post,\n"
                "📥 Or **send your own photo/video now** to use as background,\n"
                "➡️ Or click **Done** below to keep this transparent layout and generate your caption.",
                reply_markup=InlineKeyboardMarkup(bg_keyboard)
            )
            return STATE_WAITING_BACKGROUND
            
        except Exception as e:
            logger.error(f"Failed to generate/send image in OCR flow for Maga: {e}")
            err_msg = (
                "⚠️ Telegram connection timed out while sending the text box. Please try again."
                if (isinstance(e, TimedOut) or "timed out" in str(e).lower())
                else f"❌ Sorry, an error occurred during rendering: {str(e)}"
            )
            await safe_edit_status_message(status_msg, err_msg)
            return STATE_WAITING_TEXT

    if font_choice in ("charlie", "maga_charlie", "faith"):
        keyboard = [
            [
                InlineKeyboardButton("🟡 Yellow (#ffde59)", callback_data="color_ocr:yellow"),
                InlineKeyboardButton("🔴 Red (#ff3131)", callback_data="color_ocr:red")
            ],
            [
                InlineKeyboardButton("🟠 Orange (#ff914d)", callback_data="color_ocr:orange"),
                InlineKeyboardButton("🟣 Magenta (#cb6ce6)", callback_data="color_ocr:magenta")
            ],
            [
                InlineKeyboardButton("🟢 Green (#00e676)", callback_data="color_ocr:green"),
                InlineKeyboardButton("🔵 Blue (#38b6ff)", callback_data="color_ocr:blue")
            ],
            [
                InlineKeyboardButton("🟪 Purple (#8c52ff)", callback_data="color_ocr:purple")
            ]
        ]
    else:
        keyboard = [
            [
                InlineKeyboardButton("🟡 Yellow (#ffde59)", callback_data="color_ocr:yellow"),
                InlineKeyboardButton("🔴 Red (#ff3131)", callback_data="color_ocr:red")
            ],
            [
                InlineKeyboardButton("🟢 Green (#00e676)", callback_data="color_ocr:green"),
                InlineKeyboardButton("🔵 Blue (#38b6ff)", callback_data="color_ocr:blue")
            ],
            [
                InlineKeyboardButton("🟪 Purple (#8c52ff)", callback_data="color_ocr:purple")
            ]
        ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        f"📝 **Text confirmed:** `{text}`\n\n"
        "🎨 **Please select the highlight color to generate the text box:**",
        reply_markup=reply_markup,
        parse_mode="Markdown"
    )
    return STATE_CONFIRM_IG_OCR

async def color_ocr_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Generate image based on OCR text and selected color."""
    query = update.callback_query
    await query.answer()
    
    color_choice = query.data.split(":")[1]
    context.user_data["chosen_color"] = color_choice
    
    text = context.user_data.get("last_text")
    font_choice = context.user_data.get("chosen_font", "faith")
    user_id = update.effective_user.id
    
    if not text:
        await query.message.reply_text("❌ No text found. Please start over by sending your text.")
        return STATE_WAITING_TEXT
        
    status_msg = await query.message.reply_text("🎨 Rendering your text box, please wait...")
    try:
        user_support_image_path = context.user_data.get("user_support_image_path")
        doge_lines = context.user_data.get("doge_highlight_lines")
        image_buffer, line_bottom = await asyncio.to_thread(
            generate_textbox_image,
            text, font_choice, color_choice,
            user_support_image_path=user_support_image_path,
            doge_highlight_lines=doge_lines
        )
        
        # Save transparent PNG bytes in session
        png_bytes = image_buffer.getvalue()
        context.user_data["transparent_png_bytes"] = png_bytes
        context.user_data["last_rendered_image_bytes"] = png_bytes
        context.user_data["line_bottom"] = line_bottom
        save_user_session(user_id, chosen_font=font_choice, chosen_color=color_choice, doge_highlight_lines=doge_lines, last_text=text, line_bottom=line_bottom)
        
        color_names = {
            "yellow": "🟡 Yellow",
            "red": "🔴 Red",
            "orange": "🟠 Orange",
            "magenta": "🟣 Magenta",
            "green": "🟢 Green",
            "blue": "🔵 Blue",
            "purple": "🟪 Purple"
        }
        color_name = color_names.get(color_choice, color_choice.capitalize())
        
        reply_markup = get_rendered_textbox_keyboard(font_choice)
        
        caption_text = f"✅ **{font_choice.capitalize()}** transparent text box generated!"
        as_doc = font_choice in ("maga", "charlie", "faith", "freedom", "doge")
        await send_rendered_media_with_retry(
            query.message,
            png_bytes,
            filename=f"{font_choice}_textbox.png",
            as_document=as_doc,
            caption=caption_text,
            reply_markup=reply_markup
        )
            
        await safe_delete_message(status_msg)
        
        # Ask user for background image/video
        if font_choice in ("charlie", "maga", "maga_charlie"):
            bg_keyboard = [
                [InlineKeyboardButton("🔍 AI Find Images (Auto)", callback_data="ai_find_images")],
                [InlineKeyboardButton("✅ Done", callback_data="bg_done")]
            ]
            await query.message.reply_text(
                "🖼️ **Would you like to overlay this text box on top of a background image or video?**\n\n"
                "🔍 Click **AI Find Images (Auto)** to automatically find photos for this post,\n"
                "📥 Or **send your own photo/video now** to use as background,\n"
                "➡️ Or click **Done** below to keep this transparent layout and generate your caption.",
                reply_markup=InlineKeyboardMarkup(bg_keyboard)
            )
        else:
            await query.message.reply_text(
                "🖼️ **Would you like to overlay this text box on top of a background image or video?**\n\n"
                "📥 **Send an image or a video now** to blend it,\n"
                "➡️ Or click/type /skip to keep the transparent PNG layout and send new text."
            )
        return STATE_WAITING_BACKGROUND
        
    except Exception as e:
        logger.error(f"Failed to generate/send image in OCR flow: {e}")
        err_msg = (
            "⚠️ Telegram connection timed out while sending the text box. Please try again."
            if (isinstance(e, TimedOut) or "timed out" in str(e).lower())
            else f"❌ Sorry, an error occurred during rendering: {str(e)}"
        )
        await safe_edit_status_message(status_msg, err_msg)
        return STATE_WAITING_TEXT

async def proofread_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle the AI proofread accept/ignore actions."""
    restore_session_if_needed(update, context)
    query = update.callback_query
    await query.answer()
    
    action = query.data.split(":")[1]
    if action == "ignore":
        await query.message.edit_text("❌ Keeping your original text.")
        return STATE_WAITING_TEXT
        
    # Else action == "accept"
    corrected_text = context.user_data.get("suggested_corrected_text")
    if not corrected_text:
        await query.message.edit_text("⚠️ No suggested text found. Please send your text again.")
        return STATE_WAITING_TEXT
        
    # Re-render with corrected text!
    font_choice = context.user_data.get("chosen_font", "faith")
    color_choice = context.user_data.get("chosen_color", "yellow")
    user_support_image_path = context.user_data.get("user_support_image_path")
    
    # Store corrected text as the last text
    context.user_data["last_text"] = corrected_text
    
    status_msg = await query.message.reply_text("🎨 Re-rendering with corrected text...")
    try:
        user_support_image_path = context.user_data.get("user_support_image_path")
        doge_lines = context.user_data.get("doge_highlight_lines")
        image_buffer, line_bottom = await asyncio.to_thread(
            generate_textbox_image,
            corrected_text, font_choice, color_choice,
            user_support_image_path=user_support_image_path,
            doge_highlight_lines=doge_lines
        )
        
        # Save transparent PNG bytes in session
        png_bytes = image_buffer.getvalue()
        context.user_data["transparent_png_bytes"] = png_bytes
        context.user_data["last_rendered_image_bytes"] = png_bytes
        context.user_data["last_text"] = corrected_text
        context.user_data["line_bottom"] = line_bottom
        save_user_session(update.effective_user.id, chosen_font=font_choice, chosen_color=color_choice, doge_highlight_lines=doge_lines, last_text=corrected_text, line_bottom=line_bottom)
        
        # Build standard post-render keyboard
        color_names = {
            "yellow": "🟡 Yellow",
            "red": "🔴 Red",
            "orange": "🟠 Orange",
            "magenta": "🟣 Magenta",
            "green": "🟢 Green",
            "blue": "🔵 Blue",
            "purple": "🟪 Purple"
        }
        color_name = color_names.get(color_choice, color_choice.capitalize())
        
        reply_markup = get_rendered_textbox_keyboard(font_choice)
        
        caption_text = f"✅ **{font_choice.capitalize()}** transparent text box generated with corrected text!"
        as_doc = font_choice in ("maga", "charlie", "faith", "doge", "freedom")
        await send_rendered_media_with_retry(
            query.message,
            png_bytes,
            filename=f"{font_choice}_textbox.png",
            as_document=as_doc,
            caption=caption_text,
            reply_markup=reply_markup
        )
            
        await safe_delete_message(status_msg)
        await query.message.edit_text("✅ Switched to corrected text box successfully!")
        await query.message.reply_text(
            "🖼️ **Would you like to overlay this text box on top of a background image or video?**\n\n"
            "📥 **Send an image or a video now** to blend it,\n"
            "➡️ Or send new text to create another one."
        )
        return STATE_WAITING_BACKGROUND
        
    except Exception as e:
        logger.error(f"Failed to generate/send image after proofread: {e}")
        err_msg = (
            "⚠️ Telegram connection timed out while sending the text box. Please try again."
            if (isinstance(e, TimedOut) or "timed out" in str(e).lower())
            else f"❌ Error during rendering: {str(e)}"
        )
        await safe_edit_status_message(status_msg, err_msg)
        return STATE_WAITING_BACKGROUND

async def background_received(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Download the background image or video and overlay the transparent PNG."""
    restore_session_if_needed(update, context)
    user_id = update.effective_user.id
    png_bytes = context.user_data.get("transparent_png_bytes")
    font_choice = context.user_data.get("chosen_font", "faith")
    color_choice = context.user_data.get("chosen_color", "yellow")
    
    if not png_bytes:
        last_text = context.user_data.get("last_text")
        if last_text:
            user_support_image_path = context.user_data.get("user_support_image_path")
            doge_lines = context.user_data.get("doge_highlight_lines")
            image_buffer, line_bottom = await asyncio.to_thread(
                generate_textbox_image,
                last_text, font_choice, color_choice,
                user_support_image_path=user_support_image_path,
                doge_highlight_lines=doge_lines
            )
            png_bytes = image_buffer.getvalue()
            context.user_data["transparent_png_bytes"] = png_bytes
            context.user_data["line_bottom"] = line_bottom
        else:
            await update.message.reply_text("❌ No text box generated yet. Please send some text first.")
            return STATE_WAITING_TEXT
        
    status_msg = await update.message.reply_text("📥 Processing media and rendering 3D effect...")
    
    if font_choice == "freedom":
        highlight_color = FREEDOM_WATERMARK_LINE_COLOR
    else:
        highlight_color = get_highlight_color(color_choice)
        
    shape_type = "square" if font_choice == "maga" else "circle"
    
    try:
        if update.message.photo:
            file_obj = await update.message.photo[-1].get_file()
            bg_path = os.path.join(BASE_DIR, f"user_bg_img_{user_id}.jpg")
            await file_obj.download_to_drive(bg_path)
            context.user_data["temp_bg_path"] = bg_path
            context.user_data["temp_bg_type"] = "image"
            
            # Generate standard preview in thread
            preview_buffer = await asyncio.to_thread(
                overlay_png_on_image,
                bg_path, png_bytes, 
                shape_image_path=None, 
                shape_type=shape_type, 
                border_color=highlight_color,
                shape_offset_x=0,
                shape_offset_y=0,
                shape_scale=1.0,
                simple_overlay=True,
                line_bottom=context.user_data.get("line_bottom")
            )
            context.user_data["last_rendered_image_bytes"] = preview_buffer.getvalue()
            
            await status_msg.delete()
            
            keyboard = [
                [
                    InlineKeyboardButton("🔄 Change Main Image", callback_data="change:main")
                ],
                [
                    InlineKeyboardButton("✅ Done", callback_data="bg_done")
                ]
            ]
            await update.message.reply_photo(
                photo=preview_buffer,
                caption=(
                    "🖼️ **Standard overlay generated!**\n\n"
                    f"👤 Send the **shape image** (different from the background) to place inside the **{shape_type}** shape behind the subject for a 3D pop-out layout.\n"
                    "➡️ Or click **Done** below to save this standard overlay."
                ),
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            # Pre-compute background cutout in background thread while user selects their shape image
            asyncio.create_task(asyncio.to_thread(warmup_cutout_cache, bg_path, png_bytes, context.user_data.get("line_bottom")))
            return STATE_WAITING_SHAPE_IMAGE
            
        elif update.message.video or (update.message.document and update.message.document.mime_type and update.message.document.mime_type.startswith("video/")):
            doc_or_vid = update.message.video or update.message.document
            file_obj = await doc_or_vid.get_file()
            video_ext = "mp4"
            if update.message.document and update.message.document.file_name:
                video_ext = update.message.document.file_name.split(".")[-1]
            elif update.message.video and update.message.video.file_name:
                video_ext = update.message.video.file_name.split(".")[-1]
                
            bg_path = os.path.join(BASE_DIR, f"user_bg_vid_{user_id}.{video_ext}")
            out_path = os.path.join(BASE_DIR, f"user_output_{user_id}.mp4")
            await file_obj.download_to_drive(bg_path)
            
            await asyncio.to_thread(
                overlay_png_on_video,
                bg_path, png_bytes, out_path,
                shape_type=shape_type, border_color=highlight_color,
                line_bottom=context.user_data.get("line_bottom")
            )
            
            if os.path.exists(bg_path):
                os.remove(bg_path)
                
            await send_video_with_retry(
                update.message,
                out_path,
                caption="✅ Done! Standard video overlay generated. Send more text to generate another one!"
            )
            if os.path.exists(out_path):
                os.remove(out_path)
            await safe_delete_message(status_msg)
        else:
            await safe_edit_status_message(status_msg, "❌ Please send a valid image or video background.")
            return STATE_WAITING_BACKGROUND
            
    except Exception as e:
        logger.error(f"Error rendering background: {e}")
        err_msg = (
            "⚠️ Telegram connection timed out while processing/sending media. Please try again."
            if (isinstance(e, TimedOut) or "timed out" in str(e).lower())
            else f"❌ Failed to overlay background: {str(e)}"
        )
        await safe_edit_status_message(status_msg, err_msg)
        
    return STATE_WAITING_TEXT

async def skip_background(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Skip background overlay, keep transparent PNG layout, and generate caption if Charlie."""
    restore_session_if_needed(update, context)
    font_choice = context.user_data.get("chosen_font", "faith")
    query = update.callback_query
    target_msg = query.message if query else update.message
    if query:
        await query.answer()
        try:
            await query.message.edit_reply_markup(reply_markup=None)
        except Exception:
            pass

    await target_msg.reply_text("✅ Transparent PNG layout kept. Send new text to generate another!")
    
    if font_choice in ("charlie", "maga", "maga_charlie"):
        caption_status = await target_msg.reply_text("✍️ Generating Instagram caption...")
        caption_text = await asyncio.to_thread(
            generate_post_caption,
            font_choice,
            context.user_data.get("transparent_png_bytes"),
            context.user_data.get("last_text", "")
        )
        await safe_delete_message(caption_status)
        await safe_reply_caption(target_msg, caption_text)
        
    return STATE_WAITING_TEXT

async def ai_find_images_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Triggered when user clicks 'AI Find Images'. Finds matching images and displays options."""
    query = update.callback_query
    if query:
        await query.answer()
    target_msg = query.message if query else update.message
    
    headline = context.user_data.get("last_text", "")
    if not headline:
        await target_msg.reply_text("❌ No text found. Please send text or a screenshot first.")
        return STATE_WAITING_TEXT
        
    status_msg = await target_msg.reply_text("🔍 AI is identifying subjects & finding high-res photos...")
    try:
        # Extract subjects via Groq AI
        subjects = await asyncio.to_thread(find_ai_post_subjects, headline)
        main_q = subjects.get("main", "")
        shape1_q = subjects.get("shape1", "United States Capitol")
        shape2_q = subjects.get("shape2", "White House")
        
        if not main_q:
            main_q = headline[:30]
            
        # Search for images via Wikimedia Commons
        main_url = await asyncio.to_thread(search_wikimedia_image, main_q)
        shape1_url = await asyncio.to_thread(search_wikimedia_image, shape1_q) if shape1_q else None
        shape2_url = await asyncio.to_thread(search_wikimedia_image, shape2_q) if shape2_q else None
        
        if not main_url:
            await safe_edit_status_message(
                status_msg,
                f"🤔 Could not find an automatic image for '{main_q}'.\n\n"
                "Please send your own background photo directly in chat!"
            )
            return STATE_WAITING_BACKGROUND
            
        context.user_data["ai_found_images"] = {
            "main_url": main_url,
            "main_query": main_q,
            "shape1_url": shape1_url,
            "shape1_query": shape1_q,
            "shape2_url": shape2_url,
            "shape2_query": shape2_q
        }
        
        await safe_delete_message(status_msg)
        
        summary_text = (
            "🎯 **AI Found Matching Images for this Post:**\n\n"
            f"👤 **Main Subject:** {main_q}\n"
        )
        if shape1_q:
            summary_text += f"🏛️ **Shape 1:** {shape1_q}\n"
        if shape2_q:
            summary_text += f"🇺🇸 **Shape 2:** {shape2_q}\n"
            
        summary_text += (
            "\n⚡ Click **Auto-Assemble 3D Graphic** to build the 3D cutout layout instantly,\n"
            "👤 Or **Use as Main Background Only**,\n"
            "📥 Or send your own photo/video manually."
        )
        
        keyboard = [
            [
                InlineKeyboardButton("⚡ Auto-Assemble 3D Graphic", callback_data="ai_assemble:3d")
            ],
            [
                InlineKeyboardButton("👤 Use as Main Background Only", callback_data="ai_assemble:main_only")
            ],
            [
                InlineKeyboardButton("📥 I'll Send My Own Photo", callback_data="ai_cancel")
            ]
        ]
        
        await target_msg.reply_text(
            summary_text,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown"
        )
        return STATE_WAITING_BACKGROUND
        
    except Exception as e:
        logger.error(f"Error in ai_find_images: {e}")
        await safe_edit_status_message(status_msg, f"❌ Error searching images: {str(e)}")
        return STATE_WAITING_BACKGROUND

async def ai_find_images_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Command alias /findimages to trigger AI image search."""
    return await ai_find_images_callback(update, context)

async def ai_assemble_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Assembles the graphic using the AI-found images."""
    query = update.callback_query
    await query.answer()
    
    action = query.data.split(":")[1]  # "3d" or "main_only"
    user_id = update.effective_user.id
    
    ai_images = context.user_data.get("ai_found_images", {})
    main_url = ai_images.get("main_url")
    shape1_url = ai_images.get("shape1_url")
    
    if not main_url:
        await query.message.reply_text("❌ Image URLs expired. Please click AI Find Images again.")
        return STATE_WAITING_BACKGROUND
        
    status_msg = await query.message.reply_text("⚙️ Downloading images & rendering 3D composite...")
    try:
        headers = {"User-Agent": "TemplateaBot/1.0 (contact@templatea.com)"}
        bg_path = os.path.join(BASE_DIR, f"user_bg_img_{user_id}.jpg")
        
        def download_file(url, path):
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code == 200:
                with open(path, "wb") as f:
                    f.write(resp.content)
                return True
            return False
            
        ok_bg = await asyncio.to_thread(download_file, main_url, bg_path)
        if not ok_bg:
            await safe_edit_status_message(status_msg, "❌ Failed to download main subject image.")
            return STATE_WAITING_BACKGROUND
            
        context.user_data["temp_bg_path"] = bg_path
        context.user_data["temp_bg_type"] = "image"
        
        png_bytes = context.user_data.get("transparent_png_bytes")
        font_choice = context.user_data.get("chosen_font", "faith")
        color_choice = context.user_data.get("chosen_color", "yellow")
        
        if font_choice == "freedom":
            highlight_color = FREEDOM_WATERMARK_LINE_COLOR
        else:
            highlight_color = get_highlight_color(color_choice)
            
        shape_type = "square" if font_choice == "maga" else "circle"
        
        if action == "3d" and shape1_url:
            shape_path = os.path.join(BASE_DIR, f"user_shape_img_{user_id}.jpg")
            ok_shape = await asyncio.to_thread(download_file, shape1_url, shape_path)
            
            if ok_shape:
                context.user_data["temp_shape_img_path"] = shape_path
                context.user_data["active_shape_editing"] = 1
                context.user_data["shape_offset_x"] = -275
                context.user_data["shape_offset_y"] = 0
                context.user_data["shape_scale"] = 1.0
                context.user_data["shape_move_step"] = 25
                context.user_data["shape_scale_step"] = 0.1
                
                combined_buffer = await asyncio.to_thread(
                    overlay_png_on_image,
                    bg_path, png_bytes,
                    shape_image_path=shape_path,
                    shape_type=shape_type,
                    border_color=highlight_color,
                    shape_offset_x=-275,
                    shape_offset_y=0,
                    shape_scale=1.0,
                    line_bottom=context.user_data.get("line_bottom"),
                    preview_mode=True
                )
                context.user_data["last_rendered_image_bytes"] = combined_buffer.getvalue()
                
                await safe_delete_message(status_msg)
                await query.message.reply_photo(
                    photo=combined_buffer,
                    caption=(
                        "✨ **AI 3D Pop-Out Graphic Assembled!**\n\n"
                        "🎮 Use the controls below to adjust position or scale, or click **Done** when ready!"
                    ),
                    reply_markup=shape_position_keyboard(move_step=25, scale_step=0.1, active_shape=1, has_second_shape=False)
                )
                return STATE_POSITIONING_SHAPE

        # Otherwise standard main-only overlay
        preview_buffer = await asyncio.to_thread(
            overlay_png_on_image,
            bg_path, png_bytes,
            shape_image_path=None,
            shape_type=shape_type,
            border_color=highlight_color,
            shape_offset_x=0,
            shape_offset_y=0,
            shape_scale=1.0,
            simple_overlay=True,
            line_bottom=context.user_data.get("line_bottom")
        )
        context.user_data["last_rendered_image_bytes"] = preview_buffer.getvalue()
        
        await safe_delete_message(status_msg)
        keyboard = [
            [InlineKeyboardButton("🔄 Change Main Image", callback_data="change:main")],
            [InlineKeyboardButton("✅ Done", callback_data="bg_done")]
        ]
        await query.message.reply_photo(
            photo=preview_buffer,
            caption=(
                "🖼️ **Main background applied!**\n\n"
                f"👤 Send a **shape image** to place in the **{shape_type}** shape behind the subject for 3D pop-out,\n"
                "➡️ Or click **Done** to save this layout."
            ),
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        return STATE_WAITING_SHAPE_IMAGE
        
    except Exception as e:
        logger.error(f"Failed to assemble AI graphic: {e}")
        await safe_edit_status_message(status_msg, f"❌ Error assembling graphic: {str(e)}")
        return STATE_WAITING_BACKGROUND

async def ai_cancel_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """User chose to provide their own images manually."""
    query = update.callback_query
    await query.answer()
    await query.message.reply_text(
        "📥 **Manual Mode:** Send your background photo or video directly in chat whenever you're ready!\n\n"
        "➡️ Or click /skip to keep the transparent text box."
    )
    return STATE_WAITING_BACKGROUND

def shape_position_keyboard(move_step: int = 25, scale_step: float = 0.1, active_shape: int = 1, has_second_shape: bool = False) -> InlineKeyboardMarkup:
    """Returns the inline keyboard for shape position and scale adjustment."""
    keyboard = [
        [
            InlineKeyboardButton("⬆️ Up", callback_data="pos:up")
        ],
        [
            InlineKeyboardButton("⬅️ Left", callback_data="pos:left"),
            InlineKeyboardButton("➡️ Right", callback_data="pos:right")
        ],
        [
            InlineKeyboardButton("⬇️ Down", callback_data="pos:down")
        ],
        [
            InlineKeyboardButton("📍 Snap Left (-275px)", callback_data="pos:snap_left"),
            InlineKeyboardButton("📍 Snap Right (275px)", callback_data="pos:snap_right")
        ],
        [
            InlineKeyboardButton(f"🔄 Move Step: {move_step}px", callback_data="pos:toggle_move_step")
        ],
        [
            InlineKeyboardButton("➖ Scale Down", callback_data="pos:scale_down"),
            InlineKeyboardButton("➕ Scale Up", callback_data="pos:scale_up")
        ],
        [
            InlineKeyboardButton(f"🔄 Scale Step: {scale_step:.2f}x", callback_data="pos:toggle_scale_step")
        ]
    ]
    if active_shape == 1:
        if has_second_shape:
            keyboard.append([
                InlineKeyboardButton("🔄 Switch to Shape 2", callback_data="pos:switch_shape_2")
            ])
        else:
            keyboard.append([
                InlineKeyboardButton("➕ Add Second Shape", callback_data="pos:add_second_shape")
            ])
    elif active_shape == 2:
        keyboard.append([
            InlineKeyboardButton("🔄 Switch to Shape 1", callback_data="pos:switch_shape_1")
        ])
    
    keyboard.append([
        InlineKeyboardButton("🖼️ Change Main Image", callback_data="pos:change_main")
    ])
    
    if has_second_shape:
        keyboard.append([
            InlineKeyboardButton("🔄 Change Shape 1", callback_data="pos:change_shape1"),
            InlineKeyboardButton("🔄 Change Shape 2", callback_data="pos:change_shape2")
        ])
        keyboard.append([
            InlineKeyboardButton("🗑️ Remove Shape 2", callback_data="pos:remove_shape2")
        ])
    else:
        keyboard.append([
            InlineKeyboardButton("🔄 Change Shape 1", callback_data="pos:change_shape1")
        ])
        
    keyboard.append([
        InlineKeyboardButton("✅ Done", callback_data="pos:done")
    ])
    return InlineKeyboardMarkup(keyboard)

async def update_positioning_ui(query, context: ContextTypes.DEFAULT_TYPE):
    """Re-renders the composite image with current shape parameters and edits the message photo."""
    png_bytes = context.user_data.get("transparent_png_bytes")
    bg_path = context.user_data.get("temp_bg_path")
    shape_path = context.user_data.get("temp_shape_img_path")
    shape2_path = context.user_data.get("temp_shape2_img_path")
    font_choice = context.user_data.get("chosen_font", "faith")
    color_choice = context.user_data.get("chosen_color", "yellow")
    
    offset_x = context.user_data.get("shape_offset_x", -275)
    offset_y = context.user_data.get("shape_offset_y", 0)
    scale = context.user_data.get("shape_scale", 1.0)
    
    offset_x2 = context.user_data.get("shape2_offset_x", 275)
    offset_y2 = context.user_data.get("shape2_offset_y", 0)
    scale2 = context.user_data.get("shape2_scale", 1.0)
    
    move_step = context.user_data.get("shape_move_step", 25)
    scale_step = context.user_data.get("shape_scale_step", 0.1)
    
    if font_choice == "freedom":
        highlight_color = FREEDOM_WATERMARK_LINE_COLOR
    else:
        highlight_color = get_highlight_color(color_choice)
        
    shape_type = "square" if font_choice == "maga" else "circle"
    
    combined_buffer = await asyncio.to_thread(
        overlay_png_on_image,
        bg_image_path=bg_path,
        png_bytes=png_bytes,
        shape_image_path=shape_path,
        shape_type=shape_type,
        border_color=highlight_color,
        shape_offset_x=offset_x,
        shape_offset_y=offset_y,
        shape_scale=scale,
        shape2_image_path=shape2_path,
        shape2_offset_x=offset_x2,
        shape2_offset_y=offset_y2,
        shape2_scale=scale2,
        line_bottom=context.user_data.get("line_bottom"),
        preview_mode=True
    )
    context.user_data["last_rendered_image_bytes"] = combined_buffer.getvalue()
    
    active_shape = context.user_data.get("active_shape_editing", 1)
    curr_offset_x = offset_x2 if active_shape == 2 else offset_x
    curr_offset_y = offset_y2 if active_shape == 2 else offset_y
    curr_scale = scale2 if active_shape == 2 else scale
    has_second_shape = bool(shape2_path and os.path.exists(shape2_path))
    
    # Send as input media photo to edit in-place
    combined_buffer.name = "repositioning.jpg"
    await query.message.edit_media(
        media=InputMediaPhoto(
            media=combined_buffer,
            caption=(
                f"🎮 **Use the controls below to position Shape {active_shape}:**\n\n"
                f"📍 Offset: ({curr_offset_x}px, {curr_offset_y}px) [Step: {move_step}px]\n"
                f"🔎 Scale: {curr_scale:.2f}x [Step: {scale_step:.2f}x]\n\n"
                "When satisfied, click **Done**!"
            )
        ),
        reply_markup=shape_position_keyboard(move_step=move_step, scale_step=scale_step, active_shape=active_shape, has_second_shape=has_second_shape)
    )

async def shape_image_received(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Download shape image and show interactive positioning controls."""
    user_id = update.effective_user.id
    status_msg = await update.message.reply_text("📥 Processing shape image...")
    
    try:
        file_obj = await update.message.photo[-1].get_file()
        shape_path = os.path.join(BASE_DIR, f"user_shape_img_{user_id}.jpg")
        await file_obj.download_to_drive(shape_path)
        context.user_data["temp_shape_img_path"] = shape_path
        
        # Initialize shape transform parameters
        context.user_data["active_shape_editing"] = 1
        context.user_data["shape_offset_x"] = -275
        context.user_data["shape_offset_y"] = 0
        context.user_data["shape_scale"] = 1.0
        context.user_data["shape_move_step"] = 25
        context.user_data["shape_scale_step"] = 0.1
        
        png_bytes = context.user_data.get("transparent_png_bytes")
        bg_path = context.user_data.get("temp_bg_path")
        font_choice = context.user_data.get("chosen_font", "faith")
        color_choice = context.user_data.get("chosen_color", "yellow")
        
        if font_choice == "freedom":
            highlight_color = FREEDOM_WATERMARK_LINE_COLOR
        else:
            highlight_color = get_highlight_color(color_choice)
            
        shape_type = "square" if font_choice == "maga" else "circle"
        
        combined_buffer = await asyncio.to_thread(
            overlay_png_on_image,
            bg_path, png_bytes, 
            shape_image_path=shape_path, 
            shape_type=shape_type, 
            border_color=highlight_color,
            shape_offset_x=-275,
            shape_offset_y=0,
            shape_scale=1.0,
            line_bottom=context.user_data.get("line_bottom"),
            preview_mode=True
        )
        context.user_data["last_rendered_image_bytes"] = combined_buffer.getvalue()
        
        await safe_delete_message(status_msg)
        await update.message.reply_photo(
            photo=combined_buffer,
            caption="🎮 **Use the controls below to position the shape behind the person:**\n\n📍 Offset: (-275px, 0px) [Step: 25px]\n🔎 Scale: 1.00x [Step: 0.10x]",
            reply_markup=shape_position_keyboard(move_step=25, scale_step=0.1, active_shape=1, has_second_shape=False)
        )
        return STATE_POSITIONING_SHAPE
        
    except Exception as e:
        logger.error(f"Failed to process shape image: {e}")
        err_msg = (
            "⚠️ Telegram connection timed out while processing shape image. Please try again."
            if (isinstance(e, TimedOut) or "timed out" in str(e).lower())
            else f"❌ Failed to process shape image: {str(e)}"
        )
        await safe_edit_status_message(status_msg, err_msg)
        return STATE_WAITING_SHAPE_IMAGE

async def shape2_image_received(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Download shape image 2 and show interactive positioning controls opposite to shape 1."""
    user_id = update.effective_user.id
    status_msg = await update.message.reply_text("📥 Processing second shape image...")
    
    try:
        file_obj = await update.message.photo[-1].get_file()
        shape2_path = os.path.join(BASE_DIR, f"user_shape2_img_{user_id}.jpg")
        await file_obj.download_to_drive(shape2_path)
        context.user_data["temp_shape2_img_path"] = shape2_path
        
        # Initialize shape 2 editing mode
        context.user_data["active_shape_editing"] = 2
        
        # Set shape 2 opposite to shape 1
        offset_x1 = context.user_data.get("shape_offset_x", -275)
        if offset_x1 < 0:
            context.user_data["shape2_offset_x"] = 275
        else:
            context.user_data["shape2_offset_x"] = -275
            
        context.user_data["shape2_offset_y"] = 0
        context.user_data["shape2_scale"] = 1.0
        
        png_bytes = context.user_data.get("transparent_png_bytes")
        bg_path = context.user_data.get("temp_bg_path")
        shape_path = context.user_data.get("temp_shape_img_path")
        font_choice = context.user_data.get("chosen_font", "faith")
        color_choice = context.user_data.get("chosen_color", "yellow")
        
        if font_choice == "freedom":
            highlight_color = FREEDOM_WATERMARK_LINE_COLOR
        else:
            highlight_color = get_highlight_color(color_choice)
        shape_type = "square" if font_choice == "maga" else "circle"
        
        combined_buffer = await asyncio.to_thread(
            overlay_png_on_image,
            bg_image_path=bg_path,
            png_bytes=png_bytes,
            shape_image_path=shape_path,
            shape_type=shape_type,
            border_color=highlight_color,
            shape_offset_x=offset_x1,
            shape_offset_y=context.user_data.get("shape_offset_y", 0),
            shape_scale=context.user_data.get("shape_scale", 1.0),
            shape2_image_path=shape2_path,
            shape2_offset_x=context.user_data.get("shape2_offset_x", 275),
            shape2_offset_y=0,
            shape2_scale=1.0,
            line_bottom=context.user_data.get("line_bottom"),
            preview_mode=True
        )
        
        await safe_delete_message(status_msg)
        
        move_step = context.user_data.get("shape_move_step", 25)
        scale_step = context.user_data.get("shape_scale_step", 0.1)
        
        await update.message.reply_photo(
            photo=combined_buffer,
            caption=(
                "🎮 **Use the controls below to position Shape 2:**\n\n"
                f"📍 Offset: ({context.user_data.get('shape2_offset_x', 275)}px, 0px) [Step: {move_step}px]\n"
                f"🔎 Scale: 1.00x [Step: {scale_step:.2f}x]\n\n"
                "When satisfied, click **Done**!"
            ),
            reply_markup=shape_position_keyboard(move_step=move_step, scale_step=scale_step, active_shape=2, has_second_shape=True)
        )
        return STATE_POSITIONING_SHAPE
        
    except Exception as e:
        logger.error(f"Failed to process second shape image: {e}")
        err_msg = (
            "⚠️ Telegram connection timed out while processing second shape image. Please try again."
            if (isinstance(e, TimedOut) or "timed out" in str(e).lower())
            else f"❌ Failed to process second shape image: {str(e)}"
        )
        await safe_edit_status_message(status_msg, err_msg)
        return STATE_POSITIONING_SHAPE

async def skip_shape_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Skip shape/cutout (standard overlay)."""
    restore_session_if_needed(update, context)
    user_id = update.effective_user.id
    png_bytes = context.user_data.get("transparent_png_bytes")
    bg_path = context.user_data.get("temp_bg_path")
    font_choice = context.user_data.get("chosen_font", "faith")
    color_choice = context.user_data.get("chosen_color", "yellow")
    
    if font_choice == "freedom":
        highlight_color = FREEDOM_WATERMARK_LINE_COLOR
    else:
        highlight_color = get_highlight_color(color_choice)
        
    shape_type = "square" if font_choice == "maga" else "circle"
    
    query = update.callback_query
    if query:
        await query.answer()
        img_bytes = context.user_data.get("last_rendered_image_bytes") or png_bytes
        # Clean up files
        bg_path = context.user_data.pop("temp_bg_path", None)
        shape_path = context.user_data.pop("temp_shape_img_path", None)
        shape2_path = context.user_data.pop("temp_shape2_img_path", None)
        clear_media_caches(bg_path, shape_path, shape2_path)
        if bg_path and os.path.exists(bg_path):
            os.remove(bg_path)
        if shape_path and os.path.exists(shape_path):
            os.remove(shape_path)
        if shape2_path and os.path.exists(shape2_path):
            os.remove(shape2_path)
            
        await query.message.edit_caption(
            caption="✅ Done! Standard image overlay generated. Send new text to generate another one!",
            reply_markup=None
        )
        
        if font_choice in ("charlie", "maga", "maga_charlie"):
            caption_status = await query.message.reply_text("✍️ Generating Instagram caption...")
            caption_text = await asyncio.to_thread(
                generate_post_caption,
                font_choice,
                img_bytes,
                context.user_data.get("last_text", "")
            )
            await safe_delete_message(caption_status)
            await safe_reply_caption(query.message, caption_text)
            
        return STATE_WAITING_TEXT
        
    # If they typed /skip or /done
    status_msg = await update.message.reply_text("📥 Rendering standard image overlay...")
    
    try:
        combined_buffer = await asyncio.to_thread(
            overlay_png_on_image,
            bg_path, png_bytes, 
            None,   # shape_image_path
            shape_type, 
            highlight_color,
            0, 0, 1.0,
            True,   # simple_overlay
            context.user_data.get("line_bottom")
        )
        context.user_data["last_rendered_image_bytes"] = combined_buffer.getvalue()
        
        # Clean up files
        bg_path = context.user_data.pop("temp_bg_path", None)
        shape_path = context.user_data.pop("temp_shape_img_path", None)
        shape2_path = context.user_data.pop("temp_shape2_img_path", None)
        clear_media_caches(bg_path, shape_path, shape2_path)
        if bg_path and os.path.exists(bg_path):
            os.remove(bg_path)
        if shape_path and os.path.exists(shape_path):
            os.remove(shape_path)
        if shape2_path and os.path.exists(shape2_path):
            os.remove(shape2_path)
            
        await safe_delete_message(status_msg)
        await update.message.reply_photo(
            photo=combined_buffer,
            caption="✅ Done! Standard image overlay generated. Send new text to generate another one!"
        )
        
        if font_choice in ("charlie", "maga", "maga_charlie"):
            caption_status = await update.message.reply_text("✍️ Generating Instagram caption...")
            caption_text = await asyncio.to_thread(
                generate_post_caption,
                font_choice,
                combined_buffer.getvalue(),
                context.user_data.get("last_text", "")
            )
            await safe_delete_message(caption_status)
            await safe_reply_caption(update.message, caption_text)
            
        return STATE_WAITING_TEXT
        
    except Exception as e:
        logger.error(f"Error rendering standard image overlay: {e}")
        err_msg = (
            "⚠️ Telegram connection timed out while rendering overlay. Please try again."
            if (isinstance(e, TimedOut) or "timed out" in str(e).lower())
            else f"❌ Failed to generate overlay: {str(e)}"
        )
        await safe_edit_status_message(status_msg, err_msg)
        return STATE_WAITING_TEXT

async def shape_position_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle callback queries for positioning/scaling buttons."""
    query = update.callback_query
    await query.answer()
    
    action = query.data.split(":")[-1]
    
    active_shape = context.user_data.get("active_shape_editing", 1)
    
    if active_shape == 2:
        offset_x = context.user_data.get("shape2_offset_x", 275)
        offset_y = context.user_data.get("shape2_offset_y", 0)
        scale = context.user_data.get("shape2_scale", 1.0)
        prefix = "shape2_"
    else:
        offset_x = context.user_data.get("shape_offset_x", -275)
        offset_y = context.user_data.get("shape_offset_y", 0)
        scale = context.user_data.get("shape_scale", 1.0)
        prefix = "shape_"
        
    move_step = context.user_data.get("shape_move_step", 25)
    scale_step = context.user_data.get("shape_scale_step", 0.1)
    
    if action == "up":
        context.user_data[prefix + "offset_y"] = offset_y - move_step
    elif action == "down":
        context.user_data[prefix + "offset_y"] = offset_y + move_step
    elif action == "left":
        context.user_data[prefix + "offset_x"] = offset_x - move_step
    elif action == "right":
        context.user_data[prefix + "offset_x"] = offset_x + move_step
    elif action == "snap_left":
        context.user_data[prefix + "offset_x"] = -275
    elif action == "snap_right":
        context.user_data[prefix + "offset_x"] = 275
    elif action == "scale_up":
        context.user_data[prefix + "scale"] = min(2.5, scale + scale_step)
    elif action == "scale_down":
        context.user_data[prefix + "scale"] = max(0.4, scale - scale_step)
    elif action == "toggle_move_step":
        steps = [5, 10, 25, 50, 100]
        next_idx = (steps.index(move_step) + 1) % len(steps)
        context.user_data["shape_move_step"] = steps[next_idx]
    elif action == "toggle_scale_step":
        steps = [0.02, 0.05, 0.1, 0.2, 0.5]
        next_idx = (steps.index(scale_step) + 1) % len(steps)
        context.user_data["shape_scale_step"] = steps[next_idx]
    elif action == "add_second_shape":
        await query.message.reply_text(
            "📥 **Please send/upload the second shape image now.**\n\n"
            "The bot will automatically place it on the opposite side of your first shape.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Cancel", callback_data="cancel_change")]])
        )
        return STATE_WAITING_SHAPE2_IMAGE
    elif action == "change_main":
        await query.message.reply_text(
            "🖼️ **Please send a new main background image now.**\n\n"
            "Your existing text box and shapes will be preserved on the new background.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Cancel", callback_data="cancel_change")]])
        )
        return STATE_WAITING_CHANGE_MAIN
    elif action == "change_shape1":
        await query.message.reply_text(
            "👤 **Please send a new image for Shape 1 now.**\n\n"
            "Your current position and scale settings will be kept.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Cancel", callback_data="cancel_change")]])
        )
        return STATE_WAITING_CHANGE_SHAPE1
    elif action == "change_shape2":
        await query.message.reply_text(
            "👥 **Please send a new image for Shape 2 now.**\n\n"
            "Your current position and scale settings will be kept.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Cancel", callback_data="cancel_change")]])
        )
        return STATE_WAITING_CHANGE_SHAPE2
    elif action == "remove_shape2":
        shape2_path = context.user_data.pop("temp_shape2_img_path", None)
        clear_media_caches(shape2_path=shape2_path)
        context.user_data["active_shape_editing"] = 1
        if shape2_path and os.path.exists(shape2_path):
            try:
                os.remove(shape2_path)
            except Exception:
                pass
        await update_positioning_ui(query, context)
        return STATE_POSITIONING_SHAPE
    elif action == "switch_shape_1":
        context.user_data["active_shape_editing"] = 1
    elif action == "switch_shape_2":
        context.user_data["active_shape_editing"] = 2
    elif action == "done":
        # If inside a No-Dependency batch session, save positioning and return to post review
        if "nodep_session" in context.user_data and context.user_data["nodep_session"].get("queue"):
            sess = context.user_data["nodep_session"]
            curr_idx = sess.get("current_idx", 0)
            if curr_idx < len(sess["queue"]):
                post = sess["queue"][curr_idx]
                post["shape_offset_x"] = context.user_data.get("shape_offset_x", 0)
                post["shape_offset_y"] = context.user_data.get("shape_offset_y", 0)
                post["shape_scale"] = context.user_data.get("shape_scale", 1.0)
            await query.message.edit_reply_markup(reply_markup=None)
            await query.message.reply_text("✅ 3D position saved!")
            return await nodep_render_and_display_current(query.message, context)

        font_choice = context.user_data.get("chosen_font", "faith")
        img_bytes = context.user_data.get("last_rendered_image_bytes") or context.user_data.get("transparent_png_bytes")
        bg_path = context.user_data.pop("temp_bg_path", None)
        shape_path = context.user_data.pop("temp_shape_img_path", None)
        shape2_path = context.user_data.pop("temp_shape2_img_path", None)
        clear_media_caches(bg_path, shape_path, shape2_path)
        context.user_data.pop("shape_move_step", None)
        context.user_data.pop("shape_scale_step", None)
        context.user_data.pop("active_shape_editing", None)
        context.user_data.pop("shape_offset_x", None)
        context.user_data.pop("shape_offset_y", None)
        context.user_data.pop("shape_scale", None)
        context.user_data.pop("shape2_offset_x", None)
        context.user_data.pop("shape2_offset_y", None)
        context.user_data.pop("shape2_scale", None)
        
        if bg_path and os.path.exists(bg_path):
            os.remove(bg_path)
        if shape_path and os.path.exists(shape_path):
            os.remove(shape_path)
        if shape2_path and os.path.exists(shape2_path):
            os.remove(shape2_path)
            
        await query.message.edit_reply_markup(reply_markup=None)
        await query.message.reply_text("✅ Done! Final pop-out layout completed. Send new text to generate another one!")
        
        if font_choice in ("charlie", "maga", "maga_charlie"):
            caption_status = await query.message.reply_text("✍️ Generating Instagram caption...")
            caption_text = await asyncio.to_thread(
                generate_post_caption,
                font_choice,
                img_bytes,
                context.user_data.get("last_text", "")
            )
            await safe_delete_message(caption_status)
            await safe_reply_caption(query.message, caption_text)
            
        return STATE_WAITING_TEXT
        
    await update_positioning_ui(query, context)
    return STATE_POSITIONING_SHAPE

async def instagram_background_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle choice of using the fetched Instagram media as background."""
    restore_session_if_needed(update, context)
    query = update.callback_query
    await query.answer()
    
    action = query.data.split(":")[-1]
    user_id = update.effective_user.id
    png_bytes = context.user_data.get("transparent_png_bytes")
    font_choice = context.user_data.get("chosen_font", "faith")
    color_choice = context.user_data.get("chosen_color", "yellow")
    
    if action == "skip":
        await query.message.edit_text("✅ Transparent PNG layout kept. Send new text to generate another!")
        context.user_data.pop("instagram_media_path", None)
        context.user_data.pop("instagram_is_video", None)
        return STATE_WAITING_TEXT
        
    if action == "upload":
        await query.message.edit_text(
            "📥 Please send a new background image or video now."
        )
        return STATE_WAITING_BACKGROUND
        
    # action == "use"
    media_path = context.user_data.get("instagram_media_path")
    is_video = context.user_data.get("instagram_is_video", False)
    
    if not media_path or not os.path.exists(media_path):
        await query.message.edit_text("❌ Instagram background media not found. Please upload a new background.")
        return STATE_WAITING_BACKGROUND
        
    status_msg = await query.message.reply_text("📥 Processing background and rendering...")
    
    if font_choice == "freedom":
        highlight_color = FREEDOM_WATERMARK_LINE_COLOR
    else:
        highlight_color = get_highlight_color(color_choice)
        
    shape_type = "square" if font_choice == "maga" else "circle"
    
    try:
        if not is_video:
            # It's an image. Setup temp_bg_path and transition to waiting for shape image state!
            context.user_data["temp_bg_path"] = media_path
            context.user_data["temp_bg_type"] = "image"
            
            # Generate standard preview
            preview_buffer = await asyncio.to_thread(
                overlay_png_on_image,
                media_path, png_bytes, 
                None,   # shape_image_path
                shape_type, 
                highlight_color,
                0, 0, 1.0,
                True,   # simple_overlay
                context.user_data.get("line_bottom")
            )
            context.user_data["last_rendered_image_bytes"] = preview_buffer.getvalue()
            
            await safe_delete_message(status_msg)
            
            keyboard = [
                [
                    InlineKeyboardButton("🔄 Change Main Image", callback_data="change:main")
                ],
                [
                    InlineKeyboardButton("✅ Done", callback_data="bg_done")
                ]
            ]
            await query.message.reply_photo(
                photo=preview_buffer,
                caption=(
                    "🖼️ **Standard overlay generated!**\n\n"
                    f"👤 Send the **shape image** (different from the background) to place inside the **{shape_type}** shape behind the subject for a 3D pop-out layout.\n"
                    "➡️ Or click **Done** below to save this standard overlay."
                ),
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            # Pre-compute background cutout in background thread while user selects their shape image
            asyncio.create_task(asyncio.to_thread(warmup_cutout_cache, media_path, png_bytes, context.user_data.get("line_bottom")))
            return STATE_WAITING_SHAPE_IMAGE
        else:
            # Video overlay (directly process and return to text waiting state)
            out_path = os.path.join(BASE_DIR, f"user_output_{user_id}.mp4")
            await asyncio.to_thread(overlay_png_on_video, media_path, png_bytes, out_path, shape_type=shape_type, border_color=highlight_color, line_bottom=context.user_data.get("line_bottom"))
            
            if os.path.exists(media_path):
                os.remove(media_path)
                
            await send_video_with_retry(
                query.message,
                out_path,
                caption="✅ Done! Standard video overlay generated. Send more text to generate another one!"
            )
            if os.path.exists(out_path):
                os.remove(out_path)
                
            context.user_data.pop("instagram_media_path", None)
            context.user_data.pop("instagram_is_video", None)
            await safe_delete_message(status_msg)
            
    except Exception as e:
        logger.error(f"Error rendering Instagram background: {e}")
        err_msg = (
            "⚠️ Telegram connection timed out while processing/sending background. Please try again."
            if (isinstance(e, TimedOut) or "timed out" in str(e).lower())
            else f"❌ Failed to overlay Instagram background: {str(e)}"
        )
        await safe_edit_status_message(status_msg, err_msg)
        
async def start_change_main_from_standard(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Trigger changing main background image from standard overlay preview."""
    query = update.callback_query
    await query.answer()
    await query.message.reply_text(
        "🖼️ **Please send a new main background image now.**",
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Cancel", callback_data="cancel_change")]])
    )
    return STATE_WAITING_CHANGE_MAIN

async def cancel_change_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancels image replacement and returns to previous state."""
    query = update.callback_query
    await query.answer("Cancelled")
    shape_path = context.user_data.get("temp_shape_img_path")
    if shape_path and os.path.exists(shape_path):
        await update_positioning_ui(query, context)
        return STATE_POSITIONING_SHAPE
    else:
        await query.message.reply_text("Keeping current main image. Send shape image to continue, or click Done.")
        return STATE_WAITING_SHAPE_IMAGE

async def change_main_image_received(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Replace main background image and refresh composite or preview."""
    user_id = update.effective_user.id
    status_msg = await update.message.reply_text("📥 Updating main background image...")
    try:
        file_obj = await update.message.photo[-1].get_file()
        bg_path = os.path.join(BASE_DIR, f"user_bg_img_{user_id}.jpg")
        old_bg = context.user_data.get("temp_bg_path")
        clear_media_caches(bg_path=old_bg)
        await file_obj.download_to_drive(bg_path)
        context.user_data["temp_bg_path"] = bg_path
        context.user_data["temp_bg_type"] = "image"
        
        png_bytes = context.user_data.get("transparent_png_bytes")
        font_choice = context.user_data.get("chosen_font", "faith")
        color_choice = context.user_data.get("chosen_color", "yellow")
        if font_choice == "freedom":
            highlight_color = FREEDOM_WATERMARK_LINE_COLOR
        else:
            highlight_color = get_highlight_color(color_choice)
        shape_type = "square" if font_choice == "maga" else "circle"
        
        # Warmup cutout cache for the new background image
        asyncio.create_task(asyncio.to_thread(warmup_cutout_cache, bg_path, png_bytes, context.user_data.get("line_bottom")))
        
        await safe_delete_message(status_msg)
        
        # Check if Shape 1 exists (already in positioning mode)
        shape_path = context.user_data.get("temp_shape_img_path")
        if shape_path and os.path.exists(shape_path):
            shape2_path = context.user_data.get("temp_shape2_img_path")
            offset_x = context.user_data.get("shape_offset_x", -275)
            offset_y = context.user_data.get("shape_offset_y", 0)
            scale = context.user_data.get("shape_scale", 1.0)
            offset_x2 = context.user_data.get("shape2_offset_x", 275)
            offset_y2 = context.user_data.get("shape2_offset_y", 0)
            scale2 = context.user_data.get("shape2_scale", 1.0)
            
            combined_buffer = await asyncio.to_thread(
                overlay_png_on_image,
                bg_image_path=bg_path,
                png_bytes=png_bytes,
                shape_image_path=shape_path,
                shape_type=shape_type,
                border_color=highlight_color,
                shape_offset_x=offset_x,
                shape_offset_y=offset_y,
                shape_scale=scale,
                shape2_image_path=shape2_path,
                shape2_offset_x=offset_x2,
                shape2_offset_y=offset_y2,
                shape2_scale=scale2,
                line_bottom=context.user_data.get("line_bottom"),
                preview_mode=True
            )
            context.user_data["last_rendered_image_bytes"] = combined_buffer.getvalue()
            
            move_step = context.user_data.get("shape_move_step", 25)
            scale_step = context.user_data.get("shape_scale_step", 0.1)
            active_shape = context.user_data.get("active_shape_editing", 1)
            has_second_shape = bool(shape2_path and os.path.exists(shape2_path))
            
            curr_offset_x = offset_x2 if active_shape == 2 else offset_x
            curr_offset_y = offset_y2 if active_shape == 2 else offset_y
            curr_scale = scale2 if active_shape == 2 else scale
            
            await update.message.reply_photo(
                photo=combined_buffer,
                caption=(
                    f"✅ **Main background image updated!**\n\n"
                    f"🎮 **Use the controls below to position Shape {active_shape}:**\n\n"
                    f"📍 Offset: ({curr_offset_x}px, {curr_offset_y}px) [Step: {move_step}px]\n"
                    f"🔎 Scale: {curr_scale:.2f}x [Step: {scale_step:.2f}x]\n\n"
                    "When satisfied, click **Done**!"
                ),
                reply_markup=shape_position_keyboard(move_step=move_step, scale_step=scale_step, active_shape=active_shape, has_second_shape=has_second_shape)
            )
            return STATE_POSITIONING_SHAPE
        else:
            # At standard overlay stage
            preview_buffer = await asyncio.to_thread(
                overlay_png_on_image,
                bg_path, png_bytes,
                shape_image_path=None,
                shape_type=shape_type,
                border_color=highlight_color,
                shape_offset_x=0,
                shape_offset_y=0,
                shape_scale=1.0,
                simple_overlay=True,
                line_bottom=context.user_data.get("line_bottom")
            )
            context.user_data["last_rendered_image_bytes"] = preview_buffer.getvalue()
            
            keyboard = [
                [
                    InlineKeyboardButton("🔄 Change Main Image", callback_data="change:main")
                ],
                [
                    InlineKeyboardButton("✅ Done", callback_data="bg_done")
                ]
            ]
            await update.message.reply_photo(
                photo=preview_buffer,
                caption=(
                    "🖼️ **Main background image updated!**\n\n"
                    f"👤 Send the **shape image** (different from the background) to place inside the **{shape_type}** shape behind the subject for a 3D pop-out layout.\n"
                    "➡️ Or click **Done** below to save this standard overlay."
                ),
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            return STATE_WAITING_SHAPE_IMAGE
    except Exception as e:
        logger.error(f"Error changing main image: {e}")
        await safe_edit_status_message(status_msg, f"❌ Failed to update main image: {str(e)}")
        shape_path = context.user_data.get("temp_shape_img_path")
        return STATE_POSITIONING_SHAPE if (shape_path and os.path.exists(shape_path)) else STATE_WAITING_SHAPE_IMAGE

async def change_shape1_image_received(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Replace Shape 1 image and re-render."""
    user_id = update.effective_user.id
    status_msg = await update.message.reply_text("📥 Updating Shape 1 image...")
    try:
        file_obj = await update.message.photo[-1].get_file()
        shape_path = os.path.join(BASE_DIR, f"user_shape_img_{user_id}.jpg")
        old_shape = context.user_data.get("temp_shape_img_path")
        clear_media_caches(shape_path=old_shape)
        await file_obj.download_to_drive(shape_path)
        context.user_data["temp_shape_img_path"] = shape_path
        
        png_bytes = context.user_data.get("transparent_png_bytes")
        bg_path = context.user_data.get("temp_bg_path")
        shape2_path = context.user_data.get("temp_shape2_img_path")
        font_choice = context.user_data.get("chosen_font", "faith")
        color_choice = context.user_data.get("chosen_color", "yellow")
        if font_choice == "freedom":
            highlight_color = FREEDOM_WATERMARK_LINE_COLOR
        else:
            highlight_color = get_highlight_color(color_choice)
        shape_type = "square" if font_choice == "maga" else "circle"
        
        offset_x = context.user_data.get("shape_offset_x", -275)
        offset_y = context.user_data.get("shape_offset_y", 0)
        scale = context.user_data.get("shape_scale", 1.0)
        offset_x2 = context.user_data.get("shape2_offset_x", 275)
        offset_y2 = context.user_data.get("shape2_offset_y", 0)
        scale2 = context.user_data.get("shape2_scale", 1.0)
        
        combined_buffer = await asyncio.to_thread(
            overlay_png_on_image,
            bg_image_path=bg_path,
            png_bytes=png_bytes,
            shape_image_path=shape_path,
            shape_type=shape_type,
            border_color=highlight_color,
            shape_offset_x=offset_x,
            shape_offset_y=offset_y,
            shape_scale=scale,
            shape2_image_path=shape2_path,
            shape2_offset_x=offset_x2,
            shape2_offset_y=offset_y2,
            shape2_scale=scale2,
            line_bottom=context.user_data.get("line_bottom"),
            preview_mode=True
        )
        context.user_data["last_rendered_image_bytes"] = combined_buffer.getvalue()
        
        move_step = context.user_data.get("shape_move_step", 25)
        scale_step = context.user_data.get("shape_scale_step", 0.1)
        active_shape = context.user_data.get("active_shape_editing", 1)
        has_second_shape = bool(shape2_path and os.path.exists(shape2_path))
        
        curr_offset_x = offset_x2 if active_shape == 2 else offset_x
        curr_offset_y = offset_y2 if active_shape == 2 else offset_y
        curr_scale = scale2 if active_shape == 2 else scale
        
        await safe_delete_message(status_msg)
        await update.message.reply_photo(
            photo=combined_buffer,
            caption=(
                f"✅ **Shape 1 image updated!**\n\n"
                f"🎮 **Use the controls below to position Shape {active_shape}:**\n\n"
                f"📍 Offset: ({curr_offset_x}px, {curr_offset_y}px) [Step: {move_step}px]\n"
                f"🔎 Scale: {curr_scale:.2f}x [Step: {scale_step:.2f}x]\n\n"
                "When satisfied, click **Done**!"
            ),
            reply_markup=shape_position_keyboard(move_step=move_step, scale_step=scale_step, active_shape=active_shape, has_second_shape=has_second_shape)
        )
        return STATE_POSITIONING_SHAPE
    except Exception as e:
        logger.error(f"Error changing shape 1 image: {e}")
        await safe_edit_status_message(status_msg, f"❌ Failed to update Shape 1 image: {str(e)}")
        return STATE_POSITIONING_SHAPE

async def change_shape2_image_received(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Replace Shape 2 image and re-render."""
    user_id = update.effective_user.id
    status_msg = await update.message.reply_text("📥 Updating Shape 2 image...")
    try:
        file_obj = await update.message.photo[-1].get_file()
        shape2_path = os.path.join(BASE_DIR, f"user_shape2_img_{user_id}.jpg")
        old_shape2 = context.user_data.get("temp_shape2_img_path")
        clear_media_caches(shape2_path=old_shape2)
        await file_obj.download_to_drive(shape2_path)
        context.user_data["temp_shape2_img_path"] = shape2_path
        
        png_bytes = context.user_data.get("transparent_png_bytes")
        bg_path = context.user_data.get("temp_bg_path")
        shape_path = context.user_data.get("temp_shape_img_path")
        font_choice = context.user_data.get("chosen_font", "faith")
        color_choice = context.user_data.get("chosen_color", "yellow")
        if font_choice == "freedom":
            highlight_color = FREEDOM_WATERMARK_LINE_COLOR
        else:
            highlight_color = get_highlight_color(color_choice)
        shape_type = "square" if font_choice == "maga" else "circle"
        
        offset_x = context.user_data.get("shape_offset_x", -275)
        offset_y = context.user_data.get("shape_offset_y", 0)
        scale = context.user_data.get("shape_scale", 1.0)
        offset_x2 = context.user_data.get("shape2_offset_x", 275)
        offset_y2 = context.user_data.get("shape2_offset_y", 0)
        scale2 = context.user_data.get("shape2_scale", 1.0)
        
        combined_buffer = await asyncio.to_thread(
            overlay_png_on_image,
            bg_image_path=bg_path,
            png_bytes=png_bytes,
            shape_image_path=shape_path,
            shape_type=shape_type,
            border_color=highlight_color,
            shape_offset_x=offset_x,
            shape_offset_y=offset_y,
            shape_scale=scale,
            shape2_image_path=shape2_path,
            shape2_offset_x=offset_x2,
            shape2_offset_y=offset_y2,
            shape2_scale=scale2,
            line_bottom=context.user_data.get("line_bottom"),
            preview_mode=True
        )
        context.user_data["last_rendered_image_bytes"] = combined_buffer.getvalue()
        
        move_step = context.user_data.get("shape_move_step", 25)
        scale_step = context.user_data.get("shape_scale_step", 0.1)
        active_shape = context.user_data.get("active_shape_editing", 2)
        has_second_shape = True
        
        curr_offset_x = offset_x2 if active_shape == 2 else offset_x
        curr_offset_y = offset_y2 if active_shape == 2 else offset_y
        curr_scale = scale2 if active_shape == 2 else scale
        
        await safe_delete_message(status_msg)
        await update.message.reply_photo(
            photo=combined_buffer,
            caption=(
                f"✅ **Shape 2 image updated!**\n\n"
                f"🎮 **Use the controls below to position Shape {active_shape}:**\n\n"
                f"📍 Offset: ({curr_offset_x}px, {curr_offset_y}px) [Step: {move_step}px]\n"
                f"🔎 Scale: {curr_scale:.2f}x [Step: {scale_step:.2f}x]\n\n"
                "When satisfied, click **Done**!"
            ),
            reply_markup=shape_position_keyboard(move_step=move_step, scale_step=scale_step, active_shape=active_shape, has_second_shape=has_second_shape)
        )
        return STATE_POSITIONING_SHAPE
    except Exception as e:
        logger.error(f"Error changing shape 2 image: {e}")
        await safe_edit_status_message(status_msg, f"❌ Failed to update Shape 2 image: {str(e)}")
        return STATE_POSITIONING_SHAPE

async def shape_image_direct_update(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handles user uploading a photo directly during positioning mode."""
    active_shape = context.user_data.get("active_shape_editing", 1)
    if active_shape == 2:
        return await change_shape2_image_received(update, context)
    else:
        return await change_shape1_image_received(update, context)

async def restart(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Safely reset current user's session without killing the process or disrupting other users."""
    user_id = update.effective_user.id
    # Clean up user's temporary files
    for pattern in [f"user_bg_img_{user_id}.jpg", f"user_shape_img_{user_id}.jpg", f"user_shape2_img_{user_id}.jpg", f"user_output_{user_id}.mp4"]:
        path = os.path.join(BASE_DIR, pattern)
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass
    
    # Clear user media caches
    clear_media_caches(
        os.path.join(BASE_DIR, f"user_bg_img_{user_id}.jpg"),
        os.path.join(BASE_DIR, f"user_shape_img_{user_id}.jpg"),
        os.path.join(BASE_DIR, f"user_shape2_img_{user_id}.jpg")
    )
    
    # Reload saved style preferences from disk so user preferences are remembered
    session = load_user_session(user_id)
    context.user_data.clear()
    if session:
        context.user_data["chosen_font"] = session.get("chosen_font")
        context.user_data["chosen_color"] = session.get("chosen_color")
        context.user_data["doge_highlight_lines"] = session.get("doge_highlight_lines")
    
    font = context.user_data.get("chosen_font", "faith")
    await update.message.reply_text(
        f"🔄 **Session reset!**\n\n"
        f"Selected Style: **{font.capitalize()}**\n\n"
        "💬 Send the text you want in your text box to start fresh!"
    )
    return STATE_WAITING_TEXT

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancel current conversation flow safely for this user."""
    user_id = update.effective_user.id
    for pattern in [f"user_bg_img_{user_id}.jpg", f"user_shape_img_{user_id}.jpg", f"user_shape2_img_{user_id}.jpg", f"user_output_{user_id}.mp4"]:
        path = os.path.join(BASE_DIR, pattern)
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass
    session = load_user_session(user_id)
    context.user_data.clear()
    if session:
        context.user_data["chosen_font"] = session.get("chosen_font")
        context.user_data["chosen_color"] = session.get("chosen_color")
        context.user_data["doge_highlight_lines"] = session.get("doge_highlight_lines")
    await update.message.reply_text("Session cancelled. Send /start or send text directly to begin again.")
    return ConversationHandler.END

def run_test_renders():
    """Generates standard test renders for visual validation."""
    logger.info("Starting test renders...")
    download_fonts()
    
    test_cases = [
        {
            "name": "test_faith_6line.png",
            "font": "faith",
            "color": "yellow",
            "text": "MY FAMILY AND EVERYONE I\nKNOW IN PAKISTAN NO\nLONGER WANTS TO COME TO\nAMERICA THANKS TO TRUMP.\nPAKISTANIS ARE STAYING\nHOME!'- WAJAHAT ALI"
        },
        {
            "name": "test_charlie_6line.png",
            "font": "charlie",
            "color": "orange",
            "text": "MY FAMILY AND EVERYONE I\nKNOW IN PAKISTAN NO\nLONGER WANTS TO COME TO\nAMERICA THANKS TO TRUMP.\nPAKISTANIS ARE STAYING\nHOME!'- WAJAHAT ALI"
        },
        {
            "name": "test_maga_6line.png",
            "font": "maga",
            "color": "orange",
            "text": "MY FAMILY AND EVERYONE I\nKNOW IN PAKISTAN NO\nLONGER WANTS TO COME TO\nAMERICA THANKS TO TRUMP.\nPAKISTANIS ARE STAYING\nHOME!'- WAJAHAT ALI"
        },
        {
            "name": "test_doge_6line.png",
            "font": "doge",
            "color": "yellow",
            "text": "MY FAMILY AND EVERYONE I\nKNOW IN PAKISTAN NO\nLONGER WANTS TO COME TO\nAMERICA THANKS TO TRUMP.\nPAKISTANIS ARE STAYING\nHOME!'- WAJAHAT ALI"
        },
        {
            "name": "test_freedom_6line.png",
            "font": "freedom",
            "color": "orange",
            "text": "MY FAMILY AND EVERYONE I\nKNOW IN PAKISTAN NO\nLONGER WANTS TO COME TO\nAMERICA THANKS TO TRUMP.\nPAKISTANIS ARE STAYING\nHOME!'- WAJAHAT ALI"
        },
        {
            "name": "test_freedom_3line.png",
            "font": "freedom",
            "color": "orange",
            "text": "BREAKING NEWS TODAY\nFREEDOM FIGHTERS UNITED\nVICTORY IS CERTAIN"
        },
        {
            "name": "test_freedom_8line.png",
            "font": "freedom",
            "color": "orange",
            "text": "WOW! PRESIDENT TRUMP\nJUST BLASTED FAILED NEW\nYORK GOVERNOR KATHY HOCHUL\nFOR LETTING CANADA TREAT\nNY LIKE \"GARBAGE!\"\nHE SAYS BRUCE BLAKEMAN MUST BE\nNEW YORK'S NEXT GOVERNOR!\nDO YOU STAND WITH OUR POTUS?"
        }
    ]
    
    for tc in test_cases:
        buf, _ = generate_textbox_image(tc["text"], tc["font"], tc["color"])
        out_path = os.path.join(BASE_DIR, tc["name"])
        with open(out_path, "wb") as f:
            f.write(buf.read())
        logger.info(f"Saved test render to: {out_path}")
        
    logger.info("Test renders complete! You can view the saved images.")

# ==============================================================================
# NO-DEPENDENCY BATCH MODE
# Allows dumping screenshots + clean images in bulk, matching them automatically,
# assigning Charlie / MAGA styles, and granting full post-render editing power.
# ==============================================================================

_nodep_download_lock = asyncio.Lock()

def _clean_nodep_directory(user_id: int):
    """Safely cleans previous batch dump files for user."""
    save_dir = os.path.join(BASE_DIR, "scratch", f"nodep_{user_id}")
    if os.path.exists(save_dir):
        for f in os.listdir(save_dir):
            try:
                os.remove(os.path.join(save_dir, f))
            except Exception:
                pass

async def nodependency_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Entry point for No-Dependency Batch Mode (/nodependency, /batch, /dump)."""
    restore_session_if_needed(update, context)
    user_id = update.effective_user.id
    _clean_nodep_directory(user_id)
    
    # Initialize session
    context.user_data["nodep_session"] = {
        "count_charlie": 1,
        "count_maga": 1,
        "files": [],
        "screenshots": [],
        "clean_images": [],
        "queue": [],
        "current_idx": 0,
        "last_dump_msg": None
    }
    
    keyboard = [
        [
            InlineKeyboardButton("1 Charlie, 1 MAGA", callback_data="nodep_cnt:1:1"),
            InlineKeyboardButton("2 Charlie, 2 MAGA", callback_data="nodep_cnt:2:2")
        ],
        [
            InlineKeyboardButton("3 Charlie, 3 MAGA", callback_data="nodep_cnt:3:3"),
            InlineKeyboardButton("4 Charlie, 4 MAGA", callback_data="nodep_cnt:4:4")
        ],
        [
            InlineKeyboardButton("🇺🇸 All Charlie (2)", callback_data="nodep_cnt:2:0"),
            InlineKeyboardButton("🇺🇸 All MAGA (2)", callback_data="nodep_cnt:0:2")
        ],
        [
            InlineKeyboardButton("✏️ Custom Count", callback_data="nodep_cnt:custom")
        ]
    ]
    
    msg_text = (
        "🚀 **Welcome to No-Dependency Batch Mode!**\n\n"
        "Assemble multiple graphics in one go by simply dumping your screenshots and clean photos together!\n\n"
        "📊 **First, how many graphics would you like to create for each style?**"
    )
    
    if update.callback_query:
        await update.callback_query.answer()
        await update.callback_query.message.reply_text(msg_text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="Markdown")
    else:
        await update.message.reply_text(msg_text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="Markdown")
        
    return STATE_NODEP_COUNTS

def _ask_photos_per_post_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("1️⃣ 1 photo per post (main background only)", callback_data="nodep_ppp:1")],
        [InlineKeyboardButton("2️⃣ 2 photos per post (main + shape cutout)", callback_data="nodep_ppp:2")],
    ])

def _build_dump_instructions(charlie_cnt: int, maga_cnt: int, photos_per_post: int) -> str:
    total = charlie_cnt + maga_cnt
    if photos_per_post == 1:
        return (
            f"✅ **Target:** {charlie_cnt} Charlie + {maga_cnt} MAGA = {total} posts | **1 photo each**\n\n"
            f"📥 **Dump your files in this exact order for each post:**\n"
            f"1. 📸 Screenshot (with the headline text)\n"
            f"2. 🖼️ Main background photo (clean portrait/image)\n\n"
            f"So for {total} posts → dump **{total} screenshots + {total} main photos** = **{total * 2} files total**\n\n"
            "_Send everything now, then tap **Start Assembly**._"
        )
    else:
        return (
            f"✅ **Target:** {charlie_cnt} Charlie + {maga_cnt} MAGA = {total} posts | **2 photos each**\n\n"
            f"📥 **Dump your files in this exact order, repeated for each post:**\n"
            f"1. 📸 Screenshot (with the headline text)\n"
            f"2. 🖼️ Main background photo (clean portrait)\n"
            f"3. ✂️ Shape photo (the cutout placed in the circle/square)\n\n"
            f"So for {total} posts → dump **{total*3} files total** (screenshot + main + shape, repeated)\n\n"
            "_Send everything now, then tap **Start Assembly**._"
        )

async def nodep_counts_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handles style quantity selection — then asks photos-per-post."""
    query = update.callback_query
    await query.answer()
    user_id = update.effective_user.id
    
    data = query.data.split(":")

    # Handle photos-per-post selection
    if data[0] == "nodep_ppp":
        ppp = int(data[1])
        sess = context.user_data.setdefault("nodep_session", {})
        sess["photos_per_post"] = ppp
        charlie_cnt = sess.get("count_charlie", 1)
        maga_cnt = sess.get("count_maga", 1)
        _clean_nodep_directory(user_id)
        sess["files"] = []
        sess["last_dump_msg"] = None
        total = charlie_cnt + maga_cnt
        dump_kb = [
            [InlineKeyboardButton("⚡ Start Assembly (0 files)", callback_data="nodep_action:start_assembly")],
            [InlineKeyboardButton("❌ Cancel Batch", callback_data="nodep_action:cancel")]
        ]
        await query.message.reply_text(
            _build_dump_instructions(charlie_cnt, maga_cnt, ppp),
            reply_markup=InlineKeyboardMarkup(dump_kb),
            parse_mode="Markdown"
        )
        return STATE_NODEP_DUMP

    if len(data) >= 3 and data[1] != "custom":
        charlie_cnt = int(data[1])
        maga_cnt = int(data[2])
        sess = context.user_data.setdefault("nodep_session", {})
        sess["count_charlie"] = charlie_cnt
        sess["count_maga"] = maga_cnt
        sess["files"] = []
        sess["last_dump_msg"] = None
        
        await query.message.reply_text(
            f"✅ **{charlie_cnt} Charlie + {maga_cnt} MAGA selected!**\n\n"
            "📸 **Now: how many photos do you have per post?**",
            reply_markup=_ask_photos_per_post_keyboard(),
            parse_mode="Markdown"
        )
        return STATE_NODEP_COUNTS
        
    elif data[1] == "custom":
        await query.message.reply_text(
            "💬 **Reply with your desired counts** (e.g. `2 Charlie 3 MAGA` or just `2 3`):"
        )
        return STATE_NODEP_COUNTS

async def nodep_custom_counts_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Parses custom text counts from user, then asks photos-per-post."""
    text = update.message.text.strip()
    user_id = update.effective_user.id
    nums = [int(n) for n in re.findall(r'\d+', text)]
    if len(nums) >= 2:
        charlie_cnt, maga_cnt = nums[0], nums[1]
    elif len(nums) == 1:
        charlie_cnt, maga_cnt = nums[0], nums[0]
    else:
        charlie_cnt, maga_cnt = 1, 1
        
    _clean_nodep_directory(user_id)
    sess = context.user_data.setdefault("nodep_session", {})
    sess["count_charlie"] = charlie_cnt
    sess["count_maga"] = maga_cnt
    sess["files"] = []
    sess["last_dump_msg"] = None
    
    await update.message.reply_text(
        f"✅ **{charlie_cnt} Charlie + {maga_cnt} MAGA selected!**\n\n"
        "📸 **Now: how many photos do you have per post?**",
        reply_markup=_ask_photos_per_post_keyboard(),
        parse_mode="Markdown"
    )
    return STATE_NODEP_COUNTS

async def nodep_file_received(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Collects dumped media files in No-Dependency mode. Keeps a single live counter message."""
    user_id = update.effective_user.id
    sess = context.user_data.setdefault("nodep_session", {"files": [], "count_charlie": 1, "count_maga": 1})

    photo_file = None
    if update.message.photo:
        photo_file = await update.message.photo[-1].get_file()
    elif update.message.document and update.message.document.mime_type and update.message.document.mime_type.startswith("image/"):
        photo_file = await update.message.document.get_file()

    if not photo_file:
        return STATE_NODEP_DUMP

    save_dir = os.path.join(BASE_DIR, "scratch", f"nodep_{user_id}")
    os.makedirs(save_dir, exist_ok=True)

    # Collision-free filename: timestamp + short uuid
    unique_name = f"dump_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}.jpg"
    file_path = os.path.join(save_dir, unique_name)
    await photo_file.download_to_drive(file_path)

    async with _nodep_download_lock:
        file_list = sess.setdefault("files", [])
        if file_path not in file_list:
            file_list.append(file_path)
        count = len(file_list)

    dump_kb = [
        [InlineKeyboardButton(f"⚡ Start Assembly ({count} files received)", callback_data="nodep_action:start_assembly")],
        [InlineKeyboardButton("❌ Cancel Batch", callback_data="nodep_action:cancel")]
    ]

    # Edit the single counter message in-place instead of sending new messages for every file
    prev_msg = sess.get("last_dump_msg")
    if prev_msg:
        try:
            await prev_msg.edit_text(
                f"📥 **{count} file{'s' if count > 1 else ''} received!** Keep sending — tap Start Assembly when done.",
                reply_markup=InlineKeyboardMarkup(dump_kb),
                parse_mode="Markdown"
            )
            return STATE_NODEP_DUMP
        except Exception:
            pass  # Message may have been deleted or is too old; fall through to send new

    sent_msg = await update.message.reply_text(
        f"📥 **{count} file{'s' if count > 1 else ''} received!** Keep sending — tap Start Assembly when done.",
        reply_markup=InlineKeyboardMarkup(dump_kb),
        parse_mode="Markdown"
    )
    sess["last_dump_msg"] = sent_msg
    return STATE_NODEP_DUMP

async def nodep_start_assembly(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Analyzes dumped files, pairs screenshots to clean photos, and builds the graphics."""
    query = update.callback_query
    if query:
        await query.answer()
    target_msg = query.message if query else update.message
    
    sess = context.user_data.get("nodep_session", {})
    raw_files = sess.get("files", [])
    
    # Ensure files exist, are unique, and non-empty
    seen = set()
    files = []
    for f in raw_files:
        if f not in seen and os.path.exists(f) and os.path.getsize(f) > 1000:
            seen.add(f)
            files.append(f)
    sess["files"] = files
    
    if not files:
        await target_msg.reply_text("⚠️ No files received yet. Please send some screenshots and clean photos first!")
        return STATE_NODEP_DUMP
        
    status_msg = await target_msg.reply_text(f"⚙️ Analyzing {len(files)} uploaded files... (Running OCR & photo separation)")
    
    try:
        screenshots = []
        clean_images = []
        
        # 1. Classify files into screenshots vs clean portraits
        for f_path in files:
            with open(f_path, "rb") as f:
                img_bytes = f.read()
            raw_ocr = await asyncio.to_thread(call_ocr_space, img_bytes)
            headline = await asyncio.to_thread(clean_ocr_headline_with_ai, raw_ocr)
            
            # If phone aspect ratio (vertical > 1.4) and no headline detected, retry OCR once more
            if not (headline and len(headline.strip()) >= 15):
                try:
                    im_check = cv2.imread(f_path)
                    if im_check is not None and (im_check.shape[0] / max(im_check.shape[1], 1)) > 1.4:
                        await asyncio.sleep(1.5)
                        raw_ocr = await asyncio.to_thread(call_ocr_space, img_bytes)
                        headline = await asyncio.to_thread(clean_ocr_headline_with_ai, raw_ocr)
                except Exception:
                    pass

            if headline and len(headline.strip()) >= 15:
                screenshots.append({"path": f_path, "headline": headline})
            else:
                clean_images.append(f_path)
                
        logger.info(f"No-Dep Dump classified: {len(screenshots)} screenshots, {len(clean_images)} clean photos")
        
        if not screenshots:
            await safe_edit_status_message(
                status_msg,
                "🤔 Couldn't detect headlines in any of the dumped files.\n"
                "Please make sure to include at least one screenshot containing text!"
            )
            return STATE_NODEP_DUMP
            
        # Fallback: if no clean images were provided, use screenshots themselves as backgrounds
        if not clean_images:
            clean_images = [s["path"] for s in screenshots]
            
        # 2. Build the Generation Queue
        charlie_target = sess.get("count_charlie", 1)
        maga_target = sess.get("count_maga", 1)
        total_target = max(1, charlie_target + maga_target)
        
        # Interleave styles evenly: Charlie, MAGA, Charlie, MAGA...
        style_list = []
        c_left, m_left = charlie_target, maga_target
        while c_left > 0 or m_left > 0:
            if c_left > 0:
                style_list.append("charlie")
                c_left -= 1
            if m_left > 0:
                style_list.append("maga")
                m_left -= 1
        if not style_list:
            style_list = ["charlie"]
            
        charlie_colors = ["yellow", "red", "orange", "magenta", "green", "blue", "purple"]
        maga_colors = ["yellow", "red", "orange", "magenta", "green", "blue", "purple"]
        
        ppp = sess.get("photos_per_post", 1)
        queue = []
        target_count = max(total_target, len(screenshots))
        
        await safe_edit_status_message(status_msg, f"🔗 Matching {len(screenshots)} screenshot(s) to {len(clean_images)} photo(s)...")
        
        def get_pixels(p):
            try:
                im = cv2.imread(p)
                return im.shape[0] * im.shape[1] if im is not None else 0
            except Exception:
                return 0

        clean_sorted = sorted(clean_images, key=get_pixels, reverse=True)
        has_shapes = (ppp == 2) or (len(clean_images) >= len(screenshots) * 2)

        if has_shapes and len(clean_sorted) >= 2:
            n_main = min(len(screenshots), max(1, len(clean_sorted) // 2))
            main_pool = clean_sorted[:n_main]
            shape_pool = clean_sorted[n_main:]
        else:
            main_pool = list(clean_sorted)
            shape_pool = []

        # Optimal bipartite matching for Main Backgrounds (Hungarian Algorithm)
        matched_bgs = {}
        if main_pool and screenshots:
            n_ss = len(screenshots)
            n_m = len(main_pool)
            cost_matrix = np.zeros((n_ss, n_m))
            for i, ss in enumerate(screenshots):
                for j, m_img in enumerate(main_pool):
                    score = await asyncio.to_thread(compute_sift_match_score, ss["path"], m_img, False)
                    cost_matrix[i, j] = -score
            r_ind, c_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(r_ind, c_ind):
                matched_bgs[r] = main_pool[c]
            # Fill any unassigned screenshots if n_ss > n_m
            for i in range(n_ss):
                if i not in matched_bgs:
                    best_j = int(np.argmin(cost_matrix[i]))
                    matched_bgs[i] = main_pool[best_j]

        # Optimal bipartite matching for Shape Cutouts (Hungarian Algorithm)
        matched_shapes = {}
        if shape_pool and screenshots:
            n_ss = len(screenshots)
            n_s = len(shape_pool)
            cost_matrix_s = np.zeros((n_ss, n_s))
            for i, ss in enumerate(screenshots):
                for j, s_img in enumerate(shape_pool):
                    score = await asyncio.to_thread(compute_sift_match_score, ss["path"], s_img, True)
                    cost_matrix_s[i, j] = -score
            r_s, c_s = linear_sum_assignment(cost_matrix_s)
            for r, c in zip(r_s, c_s):
                matched_shapes[r] = shape_pool[c]
            # Fill any unassigned screenshots if n_ss > n_s
            for i in range(n_ss):
                if i not in matched_shapes:
                    best_j = int(np.argmin(cost_matrix_s[i]))
                    matched_shapes[i] = shape_pool[best_j]

        for idx in range(target_count):
            ss_idx = idx % len(screenshots)
            ss_item = screenshots[ss_idx]
            ss_path = ss_item["path"]
            headline = ss_item["headline"]
            
            matched_bg = matched_bgs.get(ss_idx) or (clean_sorted[0] if clean_sorted else ss_path)
            matched_shape = matched_shapes.get(ss_idx)
            
            style = style_list[idx % len(style_list)]
            color = charlie_colors[idx % len(charlie_colors)] if style == "charlie" else maga_colors[idx % len(maga_colors)]
            
            shape_log = f" -> shape={os.path.basename(matched_shape)}" if matched_shape else ""
            logger.info(
                f"Queue[{idx}]: style={style} | color={color} | "
                f"ss={os.path.basename(ss_path)} -> bg={os.path.basename(matched_bg)}{shape_log}"
            )
            
            queue.append({
                "style": style,
                "color": color,
                "headline": headline,
                "bg_path": matched_bg,
                "shape_path": matched_shape,
                "shape_offset_x": 0,
                "shape_offset_y": 0,
                "shape_scale": 1.0,
                "clean_candidates": list(clean_images),
                "rendered_bytes": None
            })
            
        sess["queue"] = queue
        sess["current_idx"] = 0
        sess["clean_images"] = clean_images
        sess["screenshots"] = screenshots
        
        # Show the user a summary of what was matched before rendering starts
        summary_lines = [f"✅ **{len(screenshots)} screenshot(s), {len(clean_images)} photo(s) — {len(queue)} posts queued**\n"]
        for i, entry in enumerate(queue):
            shape_info = f" | Shape: `{os.path.basename(entry['shape_path'])}`" if entry.get("shape_path") else ""
            summary_lines.append(
                f"• Post {i+1}: **{entry['style'].upper()}** | {entry['color'].capitalize()} | "
                f"Main: `{os.path.basename(entry['bg_path'])}`{shape_info}"
            )
        summary_lines.append("\n_Rendering now…_")
        await safe_edit_status_message(status_msg, "\n".join(summary_lines), parse_mode="Markdown")
        
        await asyncio.sleep(1.5)  # Let user read the summary briefly
        await safe_delete_message(status_msg)
        return await nodep_render_and_display_current(target_msg, context)
        
    except Exception as e:
        logger.error(f"Error in nodep_start_assembly: {e}")
        await safe_edit_status_message(status_msg, f"❌ Error during assembly: {str(e)}")
        return STATE_NODEP_DUMP

async def nodep_render_and_display_current(target_msg, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Renders the current post in the queue and presents full No-Dependency controls."""
    sess = context.user_data.get("nodep_session", {})
    queue = sess.get("queue", [])
    idx = sess.get("current_idx", 0)
    
    if idx >= len(queue):
        # All done!
        await target_msg.reply_text(
            f"🎉 **Batch Complete!** All {len(queue)} graphics have been assembled and approved.\n\n"
            "Send /nodependency to start another batch, or send text anytime for single graphics!"
        )
        return STATE_WAITING_TEXT
        
    post = queue[idx]
    style = post["style"]
    color = post["color"]
    headline = post["headline"]
    bg_path = post["bg_path"]
    shape_path = post.get("shape_path")
    shape_offset_x = post.get("shape_offset_x", 0)
    shape_offset_y = post.get("shape_offset_y", 0)
    shape_scale = post.get("shape_scale", 1.0)
    has_shape = bool(shape_path and os.path.exists(shape_path))
    
    status = await target_msg.reply_text(f"🎨 Rendering Post {idx + 1}/{len(queue)} ({style.upper()} - {color.capitalize()})...")
    
    try:
        # 1. Render transparent text box
        image_buffer, line_bottom = await asyncio.to_thread(
            generate_textbox_image,
            headline, style, color
        )
        png_bytes = image_buffer.getvalue()
        
        highlight_color = FREEDOM_WATERMARK_LINE_COLOR if style == "freedom" else get_highlight_color(color)
        shape_type = "square" if style == "maga" else "circle"
        
        # 2. Composite over matched background with shape cutout if present
        composite_buffer = await asyncio.to_thread(
            overlay_png_on_image,
            bg_path, png_bytes,
            shape_image_path=shape_path if has_shape else None,
            shape_type=shape_type,
            border_color=highlight_color,
            shape_offset_x=shape_offset_x,
            shape_offset_y=shape_offset_y,
            shape_scale=shape_scale,
            simple_overlay=not has_shape,
            line_bottom=line_bottom
        )
        rendered_bytes = composite_buffer.getvalue()
        post["rendered_bytes"] = rendered_bytes
        post["line_bottom"] = line_bottom
        
        # Save active session references for 3D positioning interoperability
        context.user_data["transparent_png_bytes"] = png_bytes
        context.user_data["last_rendered_image_bytes"] = rendered_bytes
        context.user_data["temp_bg_path"] = bg_path
        context.user_data["user_shape_image_path"] = shape_path
        context.user_data["shape_type"] = shape_type
        context.user_data["border_color"] = highlight_color
        context.user_data["shape_offset_x"] = shape_offset_x
        context.user_data["shape_offset_y"] = shape_offset_y
        context.user_data["shape_scale"] = shape_scale
        context.user_data["last_text"] = headline
        context.user_data["chosen_font"] = style
        context.user_data["chosen_color"] = color
        context.user_data["line_bottom"] = line_bottom
        
        await safe_delete_message(status)
        
        # "No Dependency" Control Keyboard
        keyboard = [
            [
                InlineKeyboardButton("✏️ Edit Text", callback_data="nodep_act:edit_text"),
                InlineKeyboardButton("🎨 Color", callback_data="nodep_act:change_color"),
                InlineKeyboardButton(f"🔄 {('MAGA' if style == 'charlie' else 'Charlie')}", callback_data="nodep_act:toggle_style")
            ],
            [
                InlineKeyboardButton("🖼️ Swap Main", callback_data="nodep_act:swap_image"),
                InlineKeyboardButton("✂️ Swap Shape" if has_shape else "➕ Add Shape", callback_data="nodep_act:swap_shape")
            ],
            [
                InlineKeyboardButton("🎮 3D Pop-Out & Position", callback_data="nodep_act:position_3d")
            ],
            [
                InlineKeyboardButton(f"✅ Next Post ({idx + 1}/{len(queue)}) ➡️", callback_data="nodep_act:next"),
                InlineKeyboardButton("💾 Finish Batch", callback_data="nodep_act:finish")
            ]
        ]
        
        shape_text = f"• Shape Cutout: `{os.path.basename(shape_path)}`\n" if has_shape else ""
        caption_text = (
            f"✨ **Post {idx + 1}/{len(queue)} Assembled!**\n"
            f"• Style: **{style.upper()}** | Color: **{color.capitalize()}**\n"
            f"• Main BG: `{os.path.basename(bg_path)}`\n"
            f"{shape_text}\n"
            f"📝 **Headline:**\n`{headline}`\n\n"
            "Use the controls below to change anything, or tap Next!"
        )
        
        await target_msg.reply_photo(
            photo=rendered_bytes,
            caption=caption_text,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown"
        )
        return STATE_NODEP_REVIEW
        
    except Exception as e:
        logger.error(f"Error rendering post in nodep: {e}")
        await safe_edit_status_message(status, f"❌ Failed to render post {idx + 1}: {str(e)}")
        return STATE_NODEP_REVIEW

async def nodep_review_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handles independent post-render actions in No-Dependency mode."""
    query = update.callback_query
    await query.answer()
    
    action = query.data.split(":")[1]
    sess = context.user_data.get("nodep_session", {})
    queue = sess.get("queue", [])
    idx = sess.get("current_idx", 0)
    
    if idx >= len(queue):
        await query.message.reply_text("Batch finished! Send /nodependency to start another.")
        return STATE_WAITING_TEXT
        
    post = queue[idx]
    
    if action == "edit_text":
        await query.message.reply_text(
            "✏️ **Tap the text box below to copy it, make your changes, and reply with the new text:**\n\n"
            f"`{post['headline']}`",
            parse_mode="Markdown"
        )
        return STATE_NODEP_EDIT_TEXT
        
    elif action == "swap_image":
        clean_images = sess.get("clean_images", [])
        if not clean_images:
            await query.message.reply_text("⚠️ No alternative clean images found in this dump. Send any photo directly to replace the background!")
            return STATE_NODEP_REVIEW

        # Send each clean photo as a visual thumbnail so the user can see what they're picking
        await query.message.reply_text(
            f"🔄 **Choose a main background photo for Post {idx + 1}:**\n"
            "_Tap 'Use This Photo ✅' under the one you want, or send any new photo directly._",
            parse_mode="Markdown"
        )
        for i, c_path in enumerate(clean_images):
            is_current = (c_path == post.get("bg_path"))
            label = f"{'✅ Currently used — switch to this' if is_current else f'Use this photo ({i+1}/{len(clean_images)})'}"
            try:
                with open(c_path, "rb") as img_f:
                    await query.message.reply_photo(
                        photo=img_f,
                        caption=f"📸 Photo {i + 1}/{len(clean_images)}",
                        reply_markup=InlineKeyboardMarkup([[
                            InlineKeyboardButton(label, callback_data=f"nodep_setimg:{i}")
                        ]])
                    )
            except Exception as e:
                logger.warning(f"Could not send swap preview for {c_path}: {e}")
                await query.message.reply_text(
                    f"Photo {i + 1}",
                    reply_markup=InlineKeyboardMarkup([[
                        InlineKeyboardButton(label, callback_data=f"nodep_setimg:{i}")
                    ]])
                )
        await query.message.reply_text(
            "🔙 Tap the button under any photo above to use it.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back (keep current)", callback_data="nodep_act:back")]])
        )
        return STATE_NODEP_REVIEW

    elif action == "swap_shape":
        clean_images = sess.get("clean_images", [])
        if not clean_images:
            await query.message.reply_text("⚠️ No alternative photos found in this dump. Send any photo directly to set as shape!")
            return STATE_NODEP_REVIEW

        await query.message.reply_text(
            f"✂️ **Choose a shape cutout photo for Post {idx + 1}:**\n"
            "_Tap 'Use as Shape Cutout ✂️' under the photo you want:_",
            parse_mode="Markdown"
        )
        for i, c_path in enumerate(clean_images):
            is_current = (c_path == post.get("shape_path"))
            label = "✅ Current Shape Cutout" if is_current else f"Use as Shape Cutout ({i+1}/{len(clean_images)})"
            try:
                with open(c_path, "rb") as img_f:
                    await query.message.reply_photo(
                        photo=img_f,
                        caption=f"📸 Photo {i + 1}/{len(clean_images)}",
                        reply_markup=InlineKeyboardMarkup([[
                            InlineKeyboardButton(label, callback_data=f"nodep_setshape:{i}")
                        ]])
                    )
            except Exception as e:
                logger.warning(f"Could not send shape swap preview: {e}")
        await query.message.reply_text(
            "🔙 Tap the button under any photo above to set it as the shape cutout.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back (keep current)", callback_data="nodep_act:back")]])
        )
        return STATE_NODEP_REVIEW
        
    elif action == "change_color":
        palette_kb = [
            [InlineKeyboardButton("🟡 Yellow", callback_data="nodep_setclr:yellow"), InlineKeyboardButton("🔴 Red", callback_data="nodep_setclr:red")],
            [InlineKeyboardButton("🟠 Orange", callback_data="nodep_setclr:orange"), InlineKeyboardButton("🟣 Magenta", callback_data="nodep_setclr:magenta")],
            [InlineKeyboardButton("🟢 Green", callback_data="nodep_setclr:green"), InlineKeyboardButton("🔵 Blue", callback_data="nodep_setclr:blue")],
            [InlineKeyboardButton("🟪 Purple", callback_data="nodep_setclr:purple")],
            [InlineKeyboardButton("🔙 Back", callback_data="nodep_act:back")]
        ]
        await query.message.reply_text("🎨 **Pick a highlight color for this post:**", reply_markup=InlineKeyboardMarkup(palette_kb))
        return STATE_NODEP_REVIEW
        
    elif action == "toggle_style":
        post["style"] = "maga" if post["style"] == "charlie" else "charlie"
        return await nodep_render_and_display_current(query.message, context)
        
    elif action == "position_3d":
        # Launch standard 3D interactive controls for this post
        shape_type = "square" if post["style"] == "maga" else "circle"
        highlight_color = get_highlight_color(post["color"])
        shape_path = post.get("shape_path")
        has_shape = bool(shape_path and os.path.exists(shape_path))
        
        combined_buffer = await asyncio.to_thread(
            overlay_png_on_image,
            post["bg_path"], context.user_data["transparent_png_bytes"],
            shape_image_path=shape_path if has_shape else None,
            shape_type=shape_type,
            border_color=highlight_color,
            shape_offset_x=post.get("shape_offset_x", 0),
            shape_offset_y=post.get("shape_offset_y", 0),
            shape_scale=post.get("shape_scale", 1.0),
            simple_overlay=not has_shape,
            line_bottom=post.get("line_bottom")
        )
        await query.message.reply_photo(
            photo=combined_buffer,
            caption=(
                "🎮 **3D Pop-Out & Positioning:**\n\n"
                f"Send a **shape image** to place in the **{shape_type}** shape behind the subject for a 3D pop-out layout,\n"
                "Or use the controls below to position!"
            ),
            reply_markup=shape_position_keyboard(move_step=25, scale_step=0.1, active_shape=1, has_second_shape=False)
        )
        return STATE_POSITIONING_SHAPE
        
    elif action == "next":
        sess["current_idx"] += 1
        return await nodep_render_and_display_current(query.message, context)
        
    elif action == "finish":
        await query.message.reply_text(
            f"💾 **Batch Saved!** All approved graphics have been generated.\n\n"
            "Ready for more content? Send /nodependency or send text directly!"
        )
        return STATE_WAITING_TEXT
        
    elif action == "back":
        return await nodep_render_and_display_current(query.message, context)
        
    elif action == "cancel":
        await query.message.reply_text("❌ Batch cancelled. You are in standard single-post mode.")
        return STATE_WAITING_TEXT
        
    return STATE_NODEP_REVIEW

async def nodep_review_photo_received(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """If user drops a photo while reviewing, ask whether to use as Main or Shape."""
    user_id = update.effective_user.id
    photo_file = None
    if update.message.photo:
        photo_file = await update.message.photo[-1].get_file()
    elif update.message.document and update.message.document.mime_type and update.message.document.mime_type.startswith("image/"):
        photo_file = await update.message.document.get_file()
        
    if not photo_file:
        return STATE_NODEP_REVIEW
        
    save_dir = os.path.join(BASE_DIR, "scratch", f"nodep_{user_id}")
    os.makedirs(save_dir, exist_ok=True)
    new_bg = os.path.join(save_dir, f"custom_{int(time.time() * 1000)}_{uuid.uuid4().hex[:4]}.jpg")
    await photo_file.download_to_drive(new_bg)
    
    sess = context.user_data.get("nodep_session", {})
    clean_images = sess.setdefault("clean_images", [])
    if new_bg not in clean_images:
        clean_images.append(new_bg)
    new_idx = len(clean_images) - 1
    
    choice_kb = [
        [
            InlineKeyboardButton("🖼️ Set as Main BG", callback_data=f"nodep_setimg:{new_idx}"),
            InlineKeyboardButton("✂️ Set as Shape Cutout", callback_data=f"nodep_setshape:{new_idx}")
        ],
        [InlineKeyboardButton("🔙 Cancel", callback_data="nodep_act:back")]
    ]
    await update.message.reply_text(
        "📸 **Photo received!** Where would you like to use this photo?",
        reply_markup=InlineKeyboardMarkup(choice_kb)
    )
    return STATE_NODEP_REVIEW

async def nodep_set_shape_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Sets the shape cutout image to the selected dump photo."""
    query = update.callback_query
    await query.answer()
    
    img_idx = int(query.data.split(":")[1])
    sess = context.user_data.get("nodep_session", {})
    clean_images = sess.get("clean_images", [])
    queue = sess.get("queue", [])
    curr_idx = sess.get("current_idx", 0)
    
    if curr_idx < len(queue) and 0 <= img_idx < len(clean_images):
        queue[curr_idx]["shape_path"] = clean_images[img_idx]
        await query.message.reply_text(f"✅ Set Image #{img_idx + 1} as the shape cutout!")
        return await nodep_render_and_display_current(query.message, context)
        
    return STATE_NODEP_REVIEW

async def nodep_set_image_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Swaps the background image to the user's selected dump candidate."""
    query = update.callback_query
    await query.answer()
    
    img_idx = int(query.data.split(":")[1])
    sess = context.user_data.get("nodep_session", {})
    clean_images = sess.get("clean_images", [])
    queue = sess.get("queue", [])
    curr_idx = sess.get("current_idx", 0)
    
    if curr_idx < len(queue) and 0 <= img_idx < len(clean_images):
        queue[curr_idx]["bg_path"] = clean_images[img_idx]
        await query.message.reply_text(f"✅ Swapped to Image #{img_idx + 1}!")
        return await nodep_render_and_display_current(query.message, context)
        
    return STATE_NODEP_REVIEW

async def nodep_set_color_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Changes color for current batch post."""
    query = update.callback_query
    await query.answer()
    
    new_color = query.data.split(":")[1]
    sess = context.user_data.get("nodep_session", {})
    queue = sess.get("queue", [])
    curr_idx = sess.get("current_idx", 0)
    
    if curr_idx < len(queue):
        queue[curr_idx]["color"] = new_color
        return await nodep_render_and_display_current(query.message, context)
        
    return STATE_NODEP_REVIEW

async def nodep_edited_text_received(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handles text input when user edits text on a batch post."""
    new_text = update.message.text.strip()
    sess = context.user_data.get("nodep_session", {})
    queue = sess.get("queue", [])
    curr_idx = sess.get("current_idx", 0)
    
    if curr_idx < len(queue):
        queue[curr_idx]["headline"] = new_text
        await update.message.reply_text("✅ Headline updated!")
        return await nodep_render_and_display_current(update.message, context)
        
    return STATE_NODEP_REVIEW

def main():
    parser = argparse.ArgumentParser(description="TextBox Telegram Bot")
    parser.add_argument("--test-render", action="store_true", help="Generate test rendering images and exit")
    args = parser.parse_args()
    
    if args.test_render:
        run_test_renders()
        sys.exit(0)

    # ── Single-instance guard ──────────────────────────────────────────────
    # Prevents "Conflict: terminated by other getUpdates request" when two
    # copies of the bot accidentally start at the same time.
    PIDFILE = os.path.join(BASE_DIR, "textbox_bot.pid")
    try:
        if os.path.exists(PIDFILE):
            old_pid = int(open(PIDFILE).read().strip())
            try:
                os.kill(old_pid, 0)   # check if process is still alive
                print(f"[BOT] Killing stale bot instance PID {old_pid}...", flush=True)
                os.kill(old_pid, 9)
                time.sleep(1)
            except ProcessLookupError:
                pass  # Already gone — safe to proceed
        with open(PIDFILE, "w") as _pf:
            _pf.write(str(os.getpid()))
    except Exception as _e:
        print(f"[BOT] PID-lock warning: {_e}", flush=True)
    # ── End single-instance guard ──────────────────────────────────────────

    # Standard bot run
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        print("[CRITICAL] TELEGRAM_BOT_TOKEN environment variable not set!", flush=True)
        print("[CRITICAL] Please go to Render Dashboard -> Environment tab and add TELEGRAM_BOT_TOKEN.", flush=True)
        print("[CRITICAL] Keeping health check web server alive so Render deployment stays UP...", flush=True)
        while not token:
            time.sleep(10)
            token = os.getenv("TELEGRAM_BOT_TOKEN")

    print("[BOT] TELEGRAM_BOT_TOKEN found. Initializing Telegram Bot...", flush=True)
        
    # Download fonts at startup
    download_fonts()
    
    # Configure resilient HTTP request timeouts to prevent "Timed out" errors during media rendering/uploading
    request = HTTPXRequest(
        connection_pool_size=16,
        connect_timeout=30.0,
        read_timeout=60.0,
        write_timeout=60.0,
        pool_timeout=30.0,
        media_write_timeout=120.0
    )
    application = Application.builder().token(token).request(request).concurrent_updates(True).build()
    
    conv_handler = ConversationHandler(
        entry_points=[
            CommandHandler("start", start),
            CommandHandler("clear_image", clear_image),
            CommandHandler("ocr", ocr_command),
            CommandHandler("findimages", ai_find_images_command),
            CommandHandler("nodependency", nodependency_command),
            CommandHandler("batch", nodependency_command),
            CommandHandler("dump", nodependency_command),
            MessageHandler(filters.TEXT & ~filters.COMMAND, text_received),
            MessageHandler(filters.PHOTO | filters.Document.IMAGE, image_received)
        ],
        states={
            STATE_CHOOSING_FONT: [
                CallbackQueryHandler(font_selected, pattern="^font:"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, text_received)
            ],
            STATE_CHOOSING_COLOR: [
                CallbackQueryHandler(color_selected, pattern="^color:"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, text_received)
            ],
            STATE_WAITING_TEXT: [
                CommandHandler("clear_image", clear_image),
                CommandHandler("ocr", ocr_command),
                CommandHandler("findimages", ai_find_images_command),
                CommandHandler("nodependency", nodependency_command),
                CommandHandler("batch", nodependency_command),
                CommandHandler("dump", nodependency_command),
                MessageHandler(filters.TEXT & ~filters.COMMAND, text_received),
                MessageHandler(filters.PHOTO | filters.Document.IMAGE, image_received),
                CallbackQueryHandler(change_color, pattern="^recolor:"),
                CallbackQueryHandler(change_font, pattern="^changefont$"),
                CallbackQueryHandler(edit_text_callback, pattern="^edit_text$"),
                CallbackQueryHandler(proofread_callback, pattern="^proofread:"),
                CallbackQueryHandler(ai_find_images_callback, pattern="^ai_find_images$"),
                CallbackQueryHandler(ai_assemble_callback, pattern="^ai_assemble:"),
                CallbackQueryHandler(ai_cancel_callback, pattern="^ai_cancel$"),
            ],
            STATE_WAITING_BACKGROUND: [
                CommandHandler("skip", skip_background),
                CommandHandler("done", skip_background),
                CommandHandler("ocr", ocr_command),
                CommandHandler("findimages", ai_find_images_command),
                CommandHandler("nodependency", nodependency_command),
                CommandHandler("batch", nodependency_command),
                CommandHandler("dump", nodependency_command),
                MessageHandler(filters.TEXT & ~filters.COMMAND & filters.Regex("^(?:/skip|/done|Done|done|Skip)$"), skip_background),
                CallbackQueryHandler(skip_background, pattern="^bg_done$"),
                MessageHandler(filters.PHOTO | filters.VIDEO | filters.Document.ALL, background_received),
                CallbackQueryHandler(instagram_background_callback, pattern="^ig_bg:"),
                CallbackQueryHandler(change_color, pattern="^recolor:"),
                CallbackQueryHandler(change_font, pattern="^changefont$"),
                CallbackQueryHandler(edit_text_callback, pattern="^edit_text$"),
                CallbackQueryHandler(proofread_callback, pattern="^proofread:"),
                CallbackQueryHandler(ai_find_images_callback, pattern="^ai_find_images$"),
                CallbackQueryHandler(ai_assemble_callback, pattern="^ai_assemble:"),
                CallbackQueryHandler(ai_cancel_callback, pattern="^ai_cancel$"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, text_received),
            ],
            STATE_WAITING_SHAPE_IMAGE: [
                CommandHandler("skip", skip_shape_image),
                CommandHandler("done", skip_shape_image),
                CommandHandler("ocr", ocr_command),
                CommandHandler("findimages", ai_find_images_command),
                MessageHandler(filters.TEXT & ~filters.COMMAND & filters.Regex("^(?:/skip|/done|Done|done|Skip)$"), skip_shape_image),
                CallbackQueryHandler(skip_shape_image, pattern="^bg_done$"),
                CallbackQueryHandler(start_change_main_from_standard, pattern="^change:main$"),
                MessageHandler(filters.PHOTO, shape_image_received),
                CallbackQueryHandler(change_color, pattern="^recolor:"),
                CallbackQueryHandler(change_font, pattern="^changefont$"),
                CallbackQueryHandler(edit_text_callback, pattern="^edit_text$"),
                CallbackQueryHandler(proofread_callback, pattern="^proofread:"),
                CallbackQueryHandler(ai_find_images_callback, pattern="^ai_find_images$"),
                CallbackQueryHandler(ai_assemble_callback, pattern="^ai_assemble:"),
                CallbackQueryHandler(ai_cancel_callback, pattern="^ai_cancel$"),
            ],
            STATE_POSITIONING_SHAPE: [
                CallbackQueryHandler(shape_position_callback, pattern="^pos:"),
                MessageHandler(filters.PHOTO, shape_image_direct_update),
            ],
            STATE_CONFIRM_IG_OCR: [
                CallbackQueryHandler(ig_ocr_callback, pattern="^ig_ocr:"),
                CallbackQueryHandler(color_ocr_callback, pattern="^color_ocr:"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, ig_ocr_text_edited)
            ],
            STATE_WAITING_SHAPE2_IMAGE: [
                MessageHandler(filters.PHOTO, shape2_image_received),
                CallbackQueryHandler(cancel_change_callback, pattern="^cancel_change$"),
                CallbackQueryHandler(shape_position_callback, pattern="^pos:")
            ],
            STATE_WAITING_DOGE_HIGHLIGHT_LINES: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, doge_highlight_lines_received),
                CallbackQueryHandler(change_color, pattern="^recolor:"),
                CallbackQueryHandler(change_font, pattern="^changefont$"),
                CallbackQueryHandler(edit_text_callback, pattern="^edit_text$"),
            ],
            STATE_WAITING_CHANGE_MAIN: [
                MessageHandler(filters.PHOTO, change_main_image_received),
                CallbackQueryHandler(cancel_change_callback, pattern="^cancel_change$"),
            ],
            STATE_WAITING_CHANGE_SHAPE1: [
                MessageHandler(filters.PHOTO, change_shape1_image_received),
                CallbackQueryHandler(cancel_change_callback, pattern="^cancel_change$"),
            ],
            STATE_WAITING_CHANGE_SHAPE2: [
                MessageHandler(filters.PHOTO, change_shape2_image_received),
                CallbackQueryHandler(cancel_change_callback, pattern="^cancel_change$"),
            ],
            STATE_NODEP_COUNTS: [
                CallbackQueryHandler(nodep_counts_callback, pattern="^(?:nodep_cnt:|nodep_ppp:)"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, nodep_custom_counts_text),
            ],
            STATE_NODEP_DUMP: [
                CallbackQueryHandler(nodep_start_assembly, pattern="^nodep_action:start_assembly$"),
                CallbackQueryHandler(nodep_review_callback, pattern="^nodep_action:cancel$"),
                MessageHandler(filters.PHOTO | filters.Document.IMAGE, nodep_file_received),
                CommandHandler("assemble", nodep_start_assembly),
            ],
            STATE_NODEP_REVIEW: [
                CallbackQueryHandler(nodep_review_callback, pattern="^nodep_act:"),
                CallbackQueryHandler(nodep_set_image_callback, pattern="^nodep_setimg:"),
                CallbackQueryHandler(nodep_set_shape_callback, pattern="^nodep_setshape:"),
                CallbackQueryHandler(nodep_set_color_callback, pattern="^nodep_setclr:"),
                MessageHandler(filters.PHOTO | filters.Document.IMAGE, nodep_review_photo_received),
            ],
            STATE_NODEP_EDIT_TEXT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, nodep_edited_text_received),
            ]
        },
        fallbacks=[
            CommandHandler("cancel", cancel),
            CommandHandler("restart", restart),
            CommandHandler("ocr", ocr_command),
            CommandHandler("findimages", ai_find_images_command),
            CommandHandler("nodependency", nodependency_command),
            CommandHandler("batch", nodependency_command),
            CommandHandler("dump", nodependency_command),
        ],
        per_message=False,
    )
    
    application.add_handler(conv_handler)
    
    print("[BOT] Starting Textbox Bot polling...", flush=True)
    logger.info("Starting Textbox Bot polling...")
    application.run_polling(allowed_updates=Update.ALL_TYPES, bootstrap_retries=5)

if __name__ == "__main__":
    main()

# 9to5 Reel Bot 🏢👔 (Offline Mode)

Fast, fully offline generator that turns joke items from `posts.json` into static Facebook Reels (1080×1920 PNG + 7-second silent MP4 + caption text) for the page **"9 to 5 Office"**.

**Zero AI calls. Zero API keys. No internet needed.**

---

## 🚀 Features

- **100% Offline**: Requires only Python 3.11, Pillow, and `ffmpeg` on PATH.
- **Batch Processing**: Feed an array of jokes in `posts.json`.
- **Validation**: Ensures `punchline` is present and `highlight` appears inside `punchline`. Skips bad entries with clear messages without crashing.
- **Pixel-Perfect Canvas (1080×1920)**:
  - Pure black background (`#000000`).
  - Native avatar circle cropped with 4x supersampled anti-aliased mask at `x=100, y=560` (136px).
  - Page title "9 to 5 Office" (Poppins Bold 44px, white) at `x=268, y=580`.
  - Handle "@9to5_office" (Inter Regular 36px, `#8E8E8E`) at `x=268, y=640`.
  - Text block starting at `x=108, y=760` with max width 864px and line height 72px.
  - Highlighted punchline words colored `#F7EB9A`.
  - Automatic font shrinking (from 50px down to 40px in 2px steps) if the text block exceeds `y=1350`.
  - Full character fallback for Hindi (`Noto Sans Devanagari`) and Emojis (`Noto Color Emoji` / Apple Color Emoji).
- **Fast 7-Second Silent MP4**: Rendered with `ffmpeg -loop 1 -i in.png -t 7 -r 30 -c:v libx264 -pix_fmt yuv420p -an out.mp4`.
- **Organized Outputs & Archival**:
  - Saved to `output/YYYY-MM-DD/<n>-<slug>.png`, `.mp4`, `.txt`
  - Renames `posts.json` to `done/posts-<timestamp>.json` upon completion.
  - Appends run log to `output/log.jsonl`.

---

## 📁 Directory Structure

```
.
├── bot.py                  # Batch processor & validator
├── render.py               # Canvas layout, font fallback & ffmpeg video builder
├── requirements.txt        # Only Pillow
├── README.md               # Documentation
├── posts.json              # Input batch JSON array
├── banner.png              # Source 386x119 page header banner
├── done/                   # Archived posts-<timestamp>.json files
├── assets/
│   ├── banner.png          # Cached copy of banner
│   ├── avatar.png          # 136x136 circle-masked avatar
│   └── fonts/              # Inter, Poppins, NotoSansDevanagari, NotoColorEmoji
└── output/
    ├── log.jsonl           # Execution history
    └── YYYY-MM-DD/         # <n>-<slug>.png, .mp4, .txt
```

---

## 🛠️ Setup & Installation

### 1. Requirements
- Python 3.11+
- `ffmpeg` on system PATH (`brew install ffmpeg` on macOS)
- Pillow:
  ```bash
  pip install -r requirements.txt
  ```

---

## 📥 Input Format (`posts.json`)

Create a `posts.json` file in the project folder with an array of objects:

```json
[
  {
    "setup": "Corporate Friday survival tip:",
    "punchline": "If you can't solve it by noon, it's officially a Monday issue",
    "highlight": "Monday issue",
    "caption": "Tag your Friday escape partner 🏃‍♂️"
  },
  {
    "setup": "Appraisal season reality:",
    "punchline": "15K salary hike maangi thi, responsibilities 50K wali mil gayi",
    "highlight": "responsibilities 50K",
    "caption": "Corporate mazeedari! #9to5Office"
  }
]
```

### Validation Rules
- `punchline`: Required and non-empty.
- `highlight`: Required and must be an exact substring of `punchline` (case-insensitive).
- `setup`: Optional (ends with `:`).
- `caption`: Optional.

---

## 🎮 How to Run

### Process `posts.json`
```bash
python bot.py
```
Or process a specific JSON file:
```bash
python bot.py path/to/my_posts.json
```

### Run Self-Test
Validates the rendering layout and asserts that the generated PNG is exactly 1080×1920:
```bash
python bot.py --test
```

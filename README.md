# ScreenText

Simple Linux tool for extracting text from screenshots.

The app lets you select an area of the screen, runs OCR on the image, and returns recognized text.

Features:

- Take screenshots using xdg-desktop-portal
- Extract text with Tesseract OCR
- Image preprocessing with OpenCV: upscaling of small text, dark themes, gradient / noisy / colorful backgrounds, colored text, mixed light and dark areas. Several variants are recognized in parallel and the most confident result is used
- Supports multiple 3 languages (en, ru, jp)


Requirements:

- Python 3
- Tesseract OCR
- GNOME / xdg-desktop-portal


Install dependencies:

Fedora:

sudo dnf install tesseract tesseract-langpack-eng tesseract-langpack-jpn tesseract-langpack-rus


Install Python packages:

`pip install -r requirements.txt`


Usage:

Give execute permission:

`chmod +x start.sh`

Run:

`./start.sh`

Notes:

Designed mainly for Linux Wayland sessions.
Tested only on fedora 44 + kernel 6.19.10-300

# external_src/robustness/corruptions.py
from __future__ import annotations
from PIL import Image, ImageFilter, ImageEnhance
import io

def jpeg(img: Image.Image, q: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(q))
    buf.seek(0)
    return Image.open(buf).convert("RGB")

def gaussian(img: Image.Image, sigma: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=float(sigma)))

def contrast(img: Image.Image, s: float) -> Image.Image:
    return ImageEnhance.Contrast(img).enhance(float(s))

def identity(img: Image.Image) -> Image.Image:
    return img

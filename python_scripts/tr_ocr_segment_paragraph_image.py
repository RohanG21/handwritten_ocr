import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


def deskew(gray):
    """Estimate page skew from ink pixels and rotate to correct it."""
    ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(ink > 0))
    if len(coords) < 100:
        return gray
    angle = cv2.minAreaRect(coords[:, ::-1].astype(np.float32))[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) < 0.3 or abs(angle) > 10:      # ignore noise / bad estimates
        return gray
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def segment_lines(pil_img, min_height=18, pad=10, ink_frac=0.08):
    """Return a list of PIL line crops, top to bottom."""
    gray = deskew(np.array(pil_img.convert("L")))
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Ink = white (255) on black, robust to uneven lighting
    ink = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 41, 15)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    # Horizontal projection profile: how much ink is in each pixel row
    profile = gaussian_filter1d(ink.sum(axis=1).astype(float), sigma=3)
    is_text = profile > ink_frac * profile.max()

    # Contiguous runs of text rows -> line bands
    bands, start = [], None
    for y, flag in enumerate(is_text):
        if flag and start is None:
            start = y
        elif not flag and start is not None:
            bands.append((start, y)); start = None
    if start is not None:
        bands.append((start, len(is_text)))

    H, W = gray.shape
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    crops = []
    for y0, y1 in bands:
        if y1 - y0 < min_height:
            continue
        cols = np.where(ink[y0:y1].sum(axis=0) > 0)[0]
        if len(cols) == 0:
            continue
        x0, x1 = max(cols[0] - pad, 0), min(cols[-1] + pad, W)
        crops.append(Image.fromarray(rgb[max(y0 - pad, 0):min(y1 + pad, H), x0:x1]))
    return crops


# ---- TrOCR ----
name = "microsoft/trocr-large-handwritten"
device = "cuda"
processor = TrOCRProcessor.from_pretrained(name)
model = VisionEncoderDecoderModel.from_pretrained(name).to(device).eval()


@torch.no_grad()
def read_page(path, batch_size=8):
    lines = segment_lines(Image.open(path))
    out = []
    for i in range(0, len(lines), batch_size):
        batch = lines[i:i + batch_size]
        px = processor(images=batch, return_tensors="pt").pixel_values.to(device)
        ids = model.generate(px, num_beams=4, max_new_tokens=128)
        out += processor.batch_decode(ids, skip_special_tokens=True)
    return out, lines

path = Path("/content/handwritten_ocr/sample_after_visit_paragraph.jpg")
text, line_imgs = read_page(path)
print("\n".join(text))

# Debugging: save the crops so you can see what the segmenter produced
for i, im in enumerate(line_imgs):
    im.save(f"line_{i:02d}.png")

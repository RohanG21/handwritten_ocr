"""
tr_ocr_segment_paragraph_images.py - crop every labelled text line out of a folder of page scans.

For each image in ``<root>/images`` the matching ``<root>/labels/<stem>.json``
is read.  Every entry of ``lines`` has a 4-point polygon (TL, TR, BR, BL) that
is a *slanted* rectangle (the synthetic scans are rotated ~4 degrees and the
neighbouring line polygons overlap).  A plain bounding-box crop would keep the
tilt and pull in big chunks of the lines above/below, so each polygon is
perspective-rectified into an upright rectangle with a homography +
``torch.nn.functional.grid_sample`` (CPU only).

Output, one PNG per line, in ``--out-dir`` (default ``<root>/line_crops``)::

    <original image stem>_line<NN>.png
    e.g.  002_austen_emma_02_p1.png  ->  002_austen_emma_02_p1_line01.png ...

The original stem is kept untouched, so its ``_p<page>`` token (when the
source name has one) is preserved.  It is NOT synthesised for single-page files
because e.g. ``016_austen_emma_16.png`` and ``016_austen_emma_16_p1.png`` are
different renders and would otherwise collide.

Also written to ``--out-dir``:
    skipped_files.txt  every image that was skipped (corrupt image, missing /
                       unreadable label, ...) with the reason, plus per-line
                       problems and recovered-label warnings
    manifest.csv       crop file -> source image, page, line number, size, text

Robustness
    * Corrupt / truncated / zero-byte images are detected (PIL ``verify`` for
      chunk CRCs, then a full decode) and skipped + logged; the run continues.
    * Label files with trailing garbage after the JSON object are recovered
      with ``raw_decode`` (and noted); unrecoverable ones are skipped + logged.

Compatibility
    Developed/tested on torch 2.14 (CPU).  Only long-stable APIs are used
    (``from_numpy``, ``arange``, ``expand``, ``grid_sample``,
    ``linalg.solve`` -> ``inverse`` fallback), no ``torch.meshgrid``,
    no torchvision, no CUDA.  ``align_corners`` and ``bicubic`` fall back
    automatically on releases that lack them.

Usage
    python crop_lines.py --root 50_labeled_syn_scans_1
    python crop_lines.py --images-dir imgs --labels-dir labs --out-dir out \
                         --pad 4 --interp bicubic --label line
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
MIN_SIDE_PX = 2.0  # polygons thinner than this are treated as degenerate

_FALLBACK_NOTES = set()  # one-time notices about missing torch features


# --------------------------------------------------------------------------- #
# Reading inputs
# --------------------------------------------------------------------------- #
def read_image(path):
    """Load an image into a float32 tensor of shape (1, C, H, W) on the CPU.

    Raises an exception for corrupt / truncated / empty files; the caller
    treats any exception as "skip this file".
    """
    path = Path(path)
    if path.stat().st_size == 0:
        raise ValueError("zero-byte file")
    # Pass 1: structural check (PNG chunk CRCs, headers).  verify() leaves the
    # file object unusable, so the image is reopened for the real decode.
    with Image.open(path) as im:
        im.verify()
    # Pass 2: full pixel decode - catches truncated pixel data.
    with Image.open(path) as im:
        im.load()
        if im.width < 1 or im.height < 1:
            raise ValueError("empty image")
        im = im.convert("L") if im.mode in ("L", "1") else im.convert("RGB")
        arr = np.array(im, dtype=np.uint8)  # np.array -> writable copy
    t = torch.from_numpy(arr).to(torch.float32)
    t = t.unsqueeze(0).unsqueeze(0) if t.dim() == 2 else t.permute(2, 0, 1).unsqueeze(0)
    return t.contiguous()


def load_labels(path):
    """Parse a label JSON. Returns (data, warning_or_None).

    Some files contain a valid JSON object followed by stray bytes from an
    older/longer write; those are recovered with raw_decode.
    """
    text = Path(path).read_text(encoding="utf-8-sig")
    warning = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data, end = json.JSONDecoder().raw_decode(text.lstrip())  # may raise
        extra = len(text.lstrip()) - end
        warning = "label JSON had %d trailing stray character(s); recovered leading object" % extra
    if not isinstance(data, dict) or not isinstance(data.get("lines"), list):
        raise ValueError("label JSON has no 'lines' list")
    return data, warning


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def order_quad(points):
    """Return the 4 points as float64 (4, 2) ordered TL, TR, BR, BL."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    s = p.sum(axis=1)
    d = p[:, 1] - p[:, 0]  # y - x
    return np.stack([p[np.argmin(s)], p[np.argmin(d)], p[np.argmax(s)], p[np.argmax(d)]])


def _solve(A, b):
    """Solve A x = b for a small dense system on any torch version."""
    linalg = getattr(torch, "linalg", None)
    if linalg is not None and hasattr(linalg, "solve"):
        return linalg.solve(A, b)
    return torch.inverse(A).mm(b)  # very old torch


def homography(quad):
    """3x3 matrix mapping the upright rectangle [0,W]x[0,H] onto ``quad``.

    Returns (H_matrix float64 tensor, W, H) where W/H are integer output sizes
    (mean of opposite edge lengths of the quad).
    """
    tl, tr, br, bl = quad
    w = 0.5 * (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl))
    h = 0.5 * (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr))
    W, H = max(1, int(round(w))), max(1, int(round(h)))
    dst = [(0.0, 0.0), (W, 0.0), (W, H), (0.0, H)]  # rectangle corners
    A, b = [], []
    for (X, Y), (x, y) in zip(dst, quad):
        A.append([X, Y, 1, 0, 0, 0, -x * X, -x * Y]); b.append(x)
        A.append([0, 0, 0, X, Y, 1, -y * X, -y * Y]); b.append(y)
    A = torch.tensor(A, dtype=torch.float64)
    b = torch.tensor(b, dtype=torch.float64).reshape(8, 1)
    h8 = _solve(A, b).reshape(8)
    Hm = torch.cat([h8, torch.ones(1, dtype=torch.float64)]).reshape(3, 3)
    return Hm, W, H


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def _grid_sample(img, grid, interp):
    """grid_sample with graceful fallbacks for older torch releases."""
    try:
        return F.grid_sample(img, grid, mode=interp, padding_mode="zeros", align_corners=False), False
    except TypeError:  # torch < 1.3 has no align_corners (and behaves as True)
        return F.grid_sample(img, grid, mode=interp, padding_mode="zeros"), True
    except (RuntimeError, NotImplementedError, ValueError):
        if interp == "bilinear":
            raise
        if "interp" not in _FALLBACK_NOTES:
            _FALLBACK_NOTES.add("interp")
            print("note: torch %s lacks '%s' grid_sample; using bilinear" % (torch.__version__, interp),
                  file=sys.stderr)
        return _grid_sample(img, grid, "bilinear")


def rectify_polygon(img, quad, bg, pad=0, interp="bicubic"):
    """Warp the quad (TL,TR,BR,BL, pixel coords) of ``img`` to an upright crop.

    img  : (1, C, H, W) float32 in [0, 255]
    bg   : (1, C, 1, 1) paper colour used for anything sampled outside the page
    pad  : extra context (px, in the rectified space) kept around the polygon
    Returns a uint8 numpy array (h, w) or (h, w, 3).
    """
    Hm, W, H = homography(quad)
    Wo, Ho = W + 2 * pad, H + 2 * pad
    # centres of the output pixels, expressed in rectangle coordinates
    xs = torch.arange(Wo, dtype=torch.float64) + 0.5 - pad
    ys = torch.arange(Ho, dtype=torch.float64) + 0.5 - pad
    X = xs.reshape(1, Wo).expand(Ho, Wo)
    Y = ys.reshape(Ho, 1).expand(Ho, Wo)
    den = Hm[2, 0] * X + Hm[2, 1] * Y + Hm[2, 2]
    sx = (Hm[0, 0] * X + Hm[0, 1] * Y + Hm[0, 2]) / den  # source pixel coords
    sy = (Hm[1, 0] * X + Hm[1, 1] * Y + Hm[1, 2]) / den

    Hs, Ws = img.shape[2], img.shape[3]
    gx = (sx + 0.5) / Ws * 2.0 - 1.0  # align_corners=False normalisation
    gy = (sy + 0.5) / Hs * 2.0 - 1.0
    grid = torch.stack([gx, gy], dim=-1).to(torch.float32).unsqueeze(0)

    # sample (img - bg) with zero padding, so off-page area becomes paper colour
    out, legacy = _grid_sample(img - bg, grid, interp)
    if legacy:  # torch < 1.3: align_corners=True semantics, rebuild the grid
        grid = torch.stack([sx / max(Ws - 1, 1) * 2 - 1, sy / max(Hs - 1, 1) * 2 - 1], -1)
        out, _ = _grid_sample(img - bg, grid.to(torch.float32).unsqueeze(0), interp)
    out = (out + bg).clamp(0, 255).round().to(torch.uint8)[0]  # (C, Ho, Wo)
    arr = out.numpy()
    return arr[0] if arr.shape[0] == 1 else arr.transpose(1, 2, 0)


def bbox_crop(img, points, bg, pad=0):
    """Fallback for polygons that are not 4-point: axis-aligned bounding box."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    Hs, Ws = img.shape[2], img.shape[3]
    x0 = max(int(np.floor(p[:, 0].min())) - pad, 0)
    y0 = max(int(np.floor(p[:, 1].min())) - pad, 0)
    x1 = min(int(np.ceil(p[:, 0].max())) + 1 + pad, Ws)
    y1 = min(int(np.ceil(p[:, 1].max())) + 1 + pad, Hs)
    if x1 - x0 < 1 or y1 - y0 < 1:
        raise ValueError("polygon outside the image")
    arr = img[0, :, y0:y1, x0:x1].round().clamp(0, 255).to(torch.uint8).numpy()
    return arr[0] if arr.shape[0] == 1 else arr.transpose(1, 2, 0)


def paper_colour(img):
    """Median pixel value per channel = the paper background."""
    c = img.shape[1]
    return img.reshape(c, -1).median(dim=1).values.reshape(1, c, 1, 1)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def process_image(img_path, lab_path, out_dir, tag, pad, interp, manifest, notes):
    """Crop all lines of one image. Returns number of crops written.

    Raises on file-level problems (corrupt image, unusable label) so the caller
    can record the skip.
    """
    img = read_image(img_path)  # raises if corrupt
    data, warn = load_labels(lab_path)  # raises if unusable
    if warn:
        notes.append("WARNING  %s: %s" % (img_path.name, warn))

    lines = data["lines"]
    if not lines:
        return 0
    bg = paper_colour(img)
    width = max(2, len(str(len(lines))))
    written = []
    for i, ln in enumerate(lines, start=1):
        name = "%s_%s%0*d.png" % (img_path.stem, tag, width, i)
        try:
            pts = ln["points"]
            if len(pts) == 4:
                arr = rectify_polygon(img, order_quad(pts), bg, pad=pad, interp=interp)
            else:
                arr = bbox_crop(img, pts, bg, pad=pad)
            if arr.shape[0] < 1 or arr.shape[1] < 1:
                raise ValueError("empty crop")
            if min(arr.shape[0], arr.shape[1]) < MIN_SIDE_PX:
                raise ValueError("degenerate polygon (%dx%d px)" % (arr.shape[1], arr.shape[0]))
        except Exception as e:  # one bad line must not lose the others
            notes.append("LINE     %s line %d: skipped (%s)" % (img_path.name, i, e))
            continue
        Image.fromarray(arr).save(out_dir / name)
        written.append(name)
        manifest.append({
            "crop_file": name, "source_image": img_path.name,
            "page": data.get("page", ""), "line": i,
            "width": arr.shape[1], "height": arr.shape[0], "text": ln.get("text", ""),
        })
    return len(written)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="folder containing images/ and labels/ (default: .)")
    ap.add_argument("--images-dir", help="override <root>/images")
    ap.add_argument("--labels-dir", help="override <root>/labels")
    ap.add_argument("--out-dir", help="output folder (default <root>/line_crops)")
    ap.add_argument("--label", default="line", help="token put before the line number (default: line)")
    ap.add_argument("--pad", type=int, default=0, help="extra context pixels around each line (default 0)")
    ap.add_argument("--interp", default="bicubic", choices=["bicubic", "bilinear", "nearest"],
                    help="resampling filter (default bicubic; falls back to bilinear if unsupported)")
    ap.add_argument("--threads", type=int, help="torch CPU threads (default: torch's choice)")
    args = ap.parse_args(argv)

    root = Path(args.root)
    images_dir = Path(args.images_dir) if args.images_dir else root / "images"
    labels_dir = Path(args.labels_dir) if args.labels_dir else root / "labels"
    out_dir = Path(args.out_dir) if args.out_dir else root / "line_crops"
    if not images_dir.is_dir():
        ap.error("images folder not found: %s" % images_dir)
    if not labels_dir.is_dir():
        ap.error("labels folder not found: %s" % labels_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.threads:
        torch.set_num_threads(args.threads)

    files = sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)
    print("torch %s | %d image(s) in %s" % (torch.__version__, len(files), images_dir))

    skipped, notes, manifest = [], [], []
    n_ok = n_crops = 0
    with torch.no_grad():
        for p in files:
            lab = labels_dir / (p.stem + ".json")
            if not lab.is_file():
                skipped.append((p.name, "no matching label file %s" % lab.name))
                continue
            try:
                n = process_image(p, lab, out_dir, args.label, args.pad, args.interp, manifest, notes)
            except Exception as e:  # corrupt image / unusable label -> skip, note, continue
                skipped.append((p.name, "%s: %s" % (type(e).__name__, str(e).splitlines()[0] if str(e) else "")))
                continue
            n_ok += 1
            n_crops += n
            print("  %-38s %2d line crop(s)" % (p.name, n))

    with open(out_dir / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["crop_file", "source_image", "page", "line", "width", "height", "text"])
        w.writeheader()
        w.writerows(manifest)

    with open(out_dir / "skipped_files.txt", "w", encoding="utf-8") as f:
        f.write("Skipped images: %d of %d\n" % (len(skipped), len(files)))
        for name, why in skipped:
            f.write("SKIPPED  %s  <- %s\n" % (name, why))
        if notes:
            f.write("\nOther notes (not skipped):\n")
            for n in notes:
                f.write(n + "\n")

    print("\n%d image(s) processed, %d line crop(s) written to %s" % (n_ok, n_crops, out_dir))
    if skipped:
        print("%d image(s) SKIPPED (details in skipped_files.txt):" % len(skipped))
        for name, why in skipped:
            print("  - %s: %s" % (name, why))
    for n in notes:
        print(n)
    return 0


if __name__ == "__main__":
    sys.exit(main())

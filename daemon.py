"""
Persistent Namsel OCR daemon process.

Loads all heavy models (sklearn classifiers, CNN predictor, numpy, cv2, etc.)
once at startup, then accepts image paths via stdin and returns OCR results
via stdout using a simple JSON-line protocol.

Protocol:
  - Startup: Prints "READY\n" to stdout after models are loaded
  - Request: One JSON line with "command" field:
      * {"image_path": "...", "page_type": "book", "recognizer": "probout"} → OCR
      * {"command": "preprocess", "input": "...", "output": "..."} → Preprocess image
      * {"command": "segment", "input": "...", "output_dir": "...", "min_gap": N} → Segment lines
      * {"command": "quit"} → Shutdown
  - Response: One JSON line: {"status": "ok", ...} or {"status": "error", ...}
"""

import sys
import json
import os
import traceback

# Auto-update subsystem (config, release lookup, download/install GUI).
try:
    from .daemon_updater import check_for_updates_async
except ImportError:
    from daemon_updater import check_for_updates_async

# Force UTF-8 output on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8', errors='replace')

# Add parent directory to path so namsel_ocr package imports work
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import heavy modules once at startup - this is the expensive part we avoid repeating
try:
    from namsel_ocr.namsel import PageRecognizer
    from namsel_ocr.config_manager import Config
except ImportError:
    from namsel import PageRecognizer
    from config_manager import Config

# Import preprocessing and segmentation logic
import cv2
import numpy as np

# Constants from preprocess_for_namsel.py
REFERENCE_CHAR_WIDTH = 14.23
GOOD_RANGE_MIN = 8.0
GOOD_RANGE_MAX = 16.0
MIN_CONTOUR_DIM = 3
MAX_CONTOUR_DIM = 100
MIN_CONTOUR_AREA = 10

# Constants from segment_lines.py
SEGMENT_MARGIN = 2

# Pixels at/below this luma are treated as near-black scan border. Archival pecha
# scans frame the paper in black; without this, Otsu keys on border-vs-paper
# instead of paper-vs-ink and the whole page collapses to one blob (char_width -1).
DARK_BORDER_THRESHOLD = 45
# An edge-connected dark region is treated as a border only if it covers at least
# this fraction of the image, so a stray glyph touching the frame is never wiped.
MIN_BORDER_AREA_FRAC = 0.01


def neutralize_dark_border(image):
    """Repaint a near-black scan border to the paper colour so Otsu binarization
    keys on paper-vs-ink. The border is identified by GEOMETRY, not luma alone: a
    near-black region CONNECTED to the frame edge and large enough to be a frame
    (>= MIN_BORDER_AREA_FRAC of the image). Interior ink (not edge-connected) and
    small edge-touching glyphs are left untouched, so this is an exact no-op for
    normal margined pages (0 pixels changed)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    dark = (gray <= DARK_BORDER_THRESHOLD).astype(np.uint8)
    if not dark.any():
        return image
    _num, labels = cv2.connectedComponents(dark)
    edge_labels = (set(labels[0]) | set(labels[-1])
                   | set(labels[:, 0]) | set(labels[:, -1]))
    edge_labels.discard(0)  # 0 = the non-dark background
    min_area = MIN_BORDER_AREA_FRAC * gray.size
    border_labels = [lbl for lbl in edge_labels if np.count_nonzero(labels == lbl) >= min_area]
    if not border_labels:
        return image  # no frame-sized dark border touches the edge -- no-op
    border = np.isin(labels, border_labels)
    bright = gray[gray > DARK_BORDER_THRESHOLD]
    paper = int(np.median(bright)) if bright.size else 255
    out = image.copy()
    out[border] = paper
    return out


def measure_character_width(image_gray):
    """Measure actual character width via OpenCV contour bounding boxes."""
    _, binary = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    widths = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if (MIN_CONTOUR_DIM <= w <= MAX_CONTOUR_DIM and
            MIN_CONTOUR_DIM <= h <= MAX_CONTOUR_DIM and
            w * h >= MIN_CONTOUR_AREA):
            widths.append(w)

    if widths:
        return float(np.mean(widths)), len(widths)
    return -1.0, 0


def preprocess_image(input_path, output_path):
    """Scale image to Namsel's optimal char width range if needed."""
    image = cv2.imread(input_path)
    if image is None:
        return {"status": "error", "error": f"Cannot read image: {input_path}"}
    image = neutralize_dark_border(image)

    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    char_width, num_contours = measure_character_width(gray)

    result = {
        "status": "ok",
        "size": f"{w}x{h}",
        "char_width": round(char_width, 1),
        "num_contours": num_contours,
    }

    # Only scale if chars are outside the good range
    if char_width > 0 and not (GOOD_RANGE_MIN <= char_width <= GOOD_RANGE_MAX):
        scale_factor = REFERENCE_CHAR_WIDTH / char_width
        scale_factor = max(0.3, min(2.5, scale_factor))

        new_w = max(50, int(w * scale_factor))
        new_h = max(30, int(h * scale_factor))
        interp = cv2.INTER_AREA if scale_factor < 1.0 else cv2.INTER_CUBIC
        image = cv2.resize(image, (new_w, new_h), interpolation=interp)
        result["action"] = "scaled"
        result["scale_factor"] = round(scale_factor, 3)
        result["new_size"] = f"{new_w}x{new_h}"
        result["new_char_width"] = round(char_width * scale_factor, 1)
    else:
        result["action"] = "passthrough"

    cv2.imwrite(output_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    return result


def find_line_bands(binary_img, min_gap=3):
    """Detect text bands in a binarized image using horizontal projection."""
    proj = binary_img.sum(axis=1)
    
    # Use a much lower threshold - even sparse text should be detected
    # 0.1% of max instead of 1% to catch faint or sparse lines
    threshold = proj.max() * 0.001 if proj.max() > 0 else 1
    in_text = proj > threshold

    bands = []
    start = None
    gap_count = 0

    for y, is_text in enumerate(in_text):
        if is_text:
            if start is None:
                start = y
            gap_count = 0
        else:
            if start is not None:
                gap_count += 1
                if gap_count >= min_gap:
                    bands.append((start, y - gap_count))
                    start = None
                    gap_count = 0

    if start is not None:
        bands.append((start, len(in_text) - 1))

    return bands


def segment_lines(input_path, output_dir, min_gap=3):
    """Segment a multi-line image into individual line bands."""
    image = cv2.imread(input_path)
    if image is None:
        return {"status": "error", "error": f"Cannot read image: {input_path}"}
    image = neutralize_dark_border(image)

    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        return {"status": "error", "error": f"Cannot create output directory: {e}"}

    H, W = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    bands = find_line_bands(binary, min_gap=min_gap)
    char_width, _ = measure_character_width(gray)

    if not bands:
        return {
            "status": "ok",
            "num_lines": 0,
            "line_files": [],
            "char_width": round(char_width, 2),
        }

    line_files = []
    for idx, (y_start, y_end) in enumerate(bands, start=1):
        y0 = max(0, y_start - SEGMENT_MARGIN)
        y1 = min(H, y_end + SEGMENT_MARGIN + 1)
        band = image[y0:y1, :]
        filename = f"line_{idx:03d}.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, band, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        line_files.append(os.path.abspath(filepath))

    return {
        "status": "ok",
        "num_lines": len(line_files),
        "line_files": line_files,
        "char_width": round(char_width, 2),
    }


def _process_photi_request(request, image_path):
    """PhotiLines_v2 line detection path of process_request."""
    conf_kwargs = {
        "page_type": request.get("page_type", "book"),
        "recognizer": request.get("recognizer", "probout"),
        "line_break_method": "line_cut",   # internal fallback (unused by photi path)
        "low_ink": False,
        "viterbi_postprocessing": False,
        "postprocess": False,
        "clear_hr": False,
        "detect_o": False,
        "force_single_line": False,
    }
    if request.get("debug_output"):
        conf_kwargs["debug_output"] = True

    conf = Config(**conf_kwargs)

    try:
        from namsel_ocr.photi_recognizer import recognize_page_photi
    except ImportError:
        from photi_recognizer import recognize_page_photi

    try:
        text = recognize_page_photi(image_path, conf)
    except Exception as e:
        return {"status": "error", "text": "", "error": str(e)}

    return {"status": "ok", "text": text}


def process_request(request):
    """Process a single OCR request and return the result dict."""
    image_path = request["image_path"]

    if not os.path.exists(image_path):
        return {"status": "error", "text": "", "error": f"File not found: {image_path}"}

    if request.get("line_break_method") == "photi":
        return _process_photi_request(request, image_path)

    conf_kwargs = {
        "page_type": request.get("page_type", "book"),
        "recognizer": request.get("recognizer", "probout"),
        "line_break_method": request.get("line_break_method", "line_cut"),
        "low_ink": False,
        "viterbi_postprocessing": False,
        "postprocess": False,
        "clear_hr": False,
        "detect_o": False,
        "force_single_line": request.get("force_single_line", False),
    }

    # Pass through optional flags
    if request.get("enable_interactive_segmentation"):
        conf_kwargs["enable_interactive_segmentation"] = True
    if request.get("debug_output"):
        conf_kwargs["debug_output"] = True

    conf = Config(**conf_kwargs)
    rec = PageRecognizer(image_path, conf=conf)
    result = rec.recognize_page(text=True)

    text = result if isinstance(result, str) else ""
    return {"status": "ok", "text": text}


def dispatch_command(request):
    """Dispatch request to appropriate handler based on command field."""
    command = request.get("command")
    
    if command == "preprocess":
        return preprocess_image(request["input"], request["output"])
    elif command == "segment":
        return segment_lines(
            request["input"],
            request["output_dir"],
            request.get("min_gap", 3)
        )
    elif command == "quit":
        return {"status": "quit"}
    else:
        # Default: OCR request (backward compatible)
        return process_request(request)


def main():
    # Keep a reference to the real stdout for protocol messages.
    # We redirect sys.stdout to stderr during OCR processing so that
    # print() calls inside namsel (PageRecognizer, etc.) don't corrupt
    # the JSON-line protocol on stdout.
    protocol_out = sys.stdout

    # Check for updates in background (non-blocking)
    import threading
    update_thread = threading.Thread(target=check_for_updates_async, daemon=True)
    update_thread.start()

    # Signal readiness after all imports are done (models loaded)
    protocol_out.write("READY\n")
    protocol_out.flush()

    # Log to stderr so Java can separate it from protocol output
    sys.stderr.write("[DAEMON] Namsel OCR daemon ready, waiting for requests...\n")
    sys.stderr.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError as e:
            response = {"status": "error", "text": "", "error": f"Invalid JSON: {e}"}
            protocol_out.write(json.dumps(response, ensure_ascii=False) + "\n")
            protocol_out.flush()
            continue

        if request.get("command") == "quit":
            sys.stderr.write("[DAEMON] Received quit command, shutting down.\n")
            sys.stderr.flush()
            break

        try:
            # Redirect stdout -> stderr while processing so namsel's
            # print() calls don't mix with protocol JSON on stdout
            sys.stdout = sys.stderr
            response = dispatch_command(request)
            
            # Check if quit was requested
            if response.get("status") == "quit":
                sys.stdout = protocol_out
                sys.stderr.write("[DAEMON] Quit command processed, shutting down.\n")
                sys.stderr.flush()
                break
                
        except Exception as e:
            sys.stderr.write(f"[DAEMON] Error processing request: {traceback.format_exc()}\n")
            sys.stderr.flush()
            response = {"status": "error", "text": "", "error": str(e)}
        finally:
            # Restore stdout for protocol output
            sys.stdout = protocol_out

        protocol_out.write(json.dumps(response, ensure_ascii=False) + "\n")
        protocol_out.flush()

    sys.stderr.write("[DAEMON] Daemon exiting.\n")
    sys.stderr.flush()


if __name__ == "__main__":
    main()

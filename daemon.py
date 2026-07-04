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


def _installer_filename(download_url):
    """Filename for the downloaded installer: URL basename, query stripped, platform extension ensured."""
    filename = download_url.split("/")[-1]
    if "?" in filename:
        filename = filename.split("?")[0]
    if not filename.endswith((".exe", ".pkg")):
        is_windows = sys.platform.startswith("win")
        filename = filename + (".exe" if is_windows else ".pkg")
    return filename


def _updater_version_file():
    """Path of version.py: next to the frozen exe / this file, else the parent folder."""
    import sys as _sys
    if getattr(_sys, 'frozen', False):
        installer_dir = os.path.dirname(_sys.executable)
    else:
        installer_dir = os.path.dirname(os.path.abspath(__file__))
    version_file = os.path.join(installer_dir, 'version.py')
    # Also check parent folder (Documents\namsel) if not found next to exe
    if not os.path.exists(version_file):
        version_file = os.path.join(os.path.dirname(installer_dir), 'version.py')
    return version_file


def _load_updater_config():
    """(CURRENT_VERSION, GITHUB_RELEASES_API, BRANCH_NAME) — defaults overridden by version.py."""
    current = "1.0.0.0"
    api = "https://glossariobudacompact.netlify.app/.netlify/functions/releases"
    branch = "Namsel"
    version_file = _updater_version_file()
    if os.path.exists(version_file):
        try:
            with open(version_file, 'r') as f:
                for line in f.read().split('\n'):
                    if line.startswith('VERSION ='):
                        current = line.split('=')[1].strip().strip('"\'')
                    elif line.startswith('GITHUB_RELEASES_API ='):
                        api = line.split('=')[1].strip().strip('"\'')
                    elif line.startswith('BRANCH_NAME ='):
                        branch = line.split('=')[1].strip().strip('"\'')
        except Exception as e:
            print(f"[daemon] version file unreadable, using defaults: {e}")
    return current, api, branch


def _fetch_branch_release(api, branch):
    """The first release dict whose target_commitish matches branch, or None."""
    import urllib.request
    import json
    req = urllib.request.Request(
        api,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "NamselOCR-AutoUpdater/1.0"
        }
    )
    with urllib.request.urlopen(req, timeout=15) as response:
        if response.getcode() != 200:
            return None
        data = json.loads(response.read().decode("utf-8"))
    releases = data if isinstance(data, list) else [data]
    for r in releases:
        if isinstance(r, dict) and r.get("target_commitish") == branch:
            return r
    return None


def _parse_version(v):
    try:
        return tuple(int(p) for p in v.split("."))
    except Exception:
        return (0, 0, 0, 0)


def _find_platform_asset_url(release):
    """browser_download_url of the first .exe (Windows) / .pkg (mac) asset, or None."""
    search_ext = ".exe" if sys.platform.startswith("win") else ".pkg"
    for asset in release.get("assets", []):
        if isinstance(asset, dict) and asset.get("name", "").lower().endswith(search_ext):
            return asset.get("browser_download_url")
    return None


def _format_release_notes(release):
    release_notes = release.get("body", "Notas de versão não disponíveis")
    if release_notes:
        import re
        release_notes = release_notes.replace("\\n", "\n")
        release_notes = re.sub(r"(?m)^[*-]\s+", "• ", release_notes)
    return release_notes


def check_for_updates_async():
    """Check for updates in background thread and show GUI if update available."""
    try:
        current_version, api, branch = _load_updater_config()
        release = _fetch_branch_release(api, branch)
        if not release:
            return
        tag_name = release.get("tag_name", "")
        if not tag_name:
            return
        latest_version = tag_name[1:] if tag_name.startswith("v") else tag_name
        if _parse_version(latest_version) <= _parse_version(current_version):
            return  # No update needed
        download_url = _find_platform_asset_url(release)
        if not download_url:
            return
        show_update_gui(latest_version, download_url, _format_release_notes(release))
    except Exception as e:
        print(f"[daemon] update check skipped: {e}")  # update check is optional


def show_update_gui(version, download_url, release_notes):
    """Show update GUI in main thread."""
    try:
        import tkinter as tk
        from tkinter import ttk, messagebox, scrolledtext
        import threading
        import urllib.request
        import tempfile
        import shutil
        from pathlib import Path
        import subprocess
        
        root = tk.Tk()
        root.title("Gerenciador de Atualizações - Namsel OCR")
        root.geometry("600x550")
        root.resizable(False, False)
        root.configure(bg="#f5f5f5")
        
        # Center window
        root.update_idletasks()
        x = (root.winfo_screenwidth() - 600) // 2
        y = (root.winfo_screenheight() - 550) // 2
        root.geometry(f"600x550+{x}+{y}")
        
        # Header
        header = tk.Frame(root, bg="#2c3e50", height=70)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        tk.Label(header, text="  Gerenciador de Atualizações - Namsel OCR",
                 bg="#2c3e50", fg="white",
                 font=("Segoe UI", 14, "bold")).pack(side=tk.LEFT, padx=20, pady=20)
        
        # Body
        body = tk.Frame(root, bg="#f5f5f5", padx=24, pady=20)
        body.pack(fill=tk.BOTH, expand=True)
        
        # Version info
        version_frame = tk.Frame(body, bg="#f5f5f5")
        version_frame.pack(fill=tk.X, pady=(0, 16))
        
        tk.Label(version_frame, text="Atualização disponível:",
                 bg="#f5f5f5", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT)
        tk.Label(version_frame, text=f"v{version}",
                 bg="#f5f5f5", fg="#27ae60", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT, padx=(8, 0))
        
        # Release notes
        tk.Label(body, text="Novidades:",
                 bg="#f5f5f5", font=("Segoe UI", 10, "bold")).pack(anchor=tk.W, pady=(0, 8))
        
        notes_text = scrolledtext.ScrolledText(
            body, height=12, width=65,
            font=("Segoe UI", 9), bg="white", fg="#333",
            relief=tk.FLAT, borderwidth=1, wrap=tk.WORD, state=tk.DISABLED
        )
        notes_text.pack(fill=tk.BOTH, expand=True)
        notes_text.config(state=tk.NORMAL)
        notes_text.insert(1.0, release_notes)
        notes_text.config(state=tk.DISABLED)
        
        # Progress frame
        progress_frame = tk.Frame(body, bg="#f5f5f5", height=60)
        progress_frame.pack(fill=tk.X, pady=(12, 0))
        progress_frame.pack_propagate(False)
        
        progress_label = tk.Label(progress_frame, text="", bg="#f5f5f5", font=("Segoe UI", 9), fg="#555")
        progress_label.pack(anchor=tk.W, pady=(0, 4))
        
        progress_bar = ttk.Progressbar(progress_frame, mode="determinate", length=540)
        
        # Buttons
        btn_frame = tk.Frame(root, bg="#ececec", pady=14)
        btn_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        close_btn = tk.Button(btn_frame, text="Fechar", width=12,
                              command=root.destroy, font=("Segoe UI", 9))
        close_btn.pack(side=tk.RIGHT, padx=(0, 20))
        
        action_btn = tk.Button(btn_frame, text="Baixar Atualização", width=18,
                               bg="#27ae60", fg="white", activebackground="#229954",
                               font=("Segoe UI", 9, "bold"), relief=tk.FLAT, cursor="hand2")
        action_btn.pack(side=tk.RIGHT, padx=(0, 8))
        
        # Download logic
        installer_path = [None]  # Mutable container
        
        def update_progress(percent):
            progress_bar['value'] = percent
            progress_label.config(text=f"Baixando... {percent}%")
        
        def do_download():
            try:
                filename = _installer_filename(download_url)

                # Download to temp
                temp_dir = Path(tempfile.mkdtemp(prefix="namsel-update-"))
                temp_file = temp_dir / filename
                
                req = urllib.request.Request(download_url, headers={"User-Agent": "NamselOCR-AutoUpdater/1.0"})
                
                with urllib.request.urlopen(req, timeout=60) as response:
                    file_size = int(response.headers.get("Content-Length", 0))
                    downloaded = 0
                    chunk_size = 8192
                    
                    with open(temp_file, "wb") as f:
                        while True:
                            chunk = response.read(chunk_size)
                            if not chunk:
                                break
                            f.write(chunk)
                            downloaded += len(chunk)
                            if file_size > 0:
                                progress = int((downloaded * 100) / file_size)
                                root.after(0, lambda p=progress: update_progress(p))
                
                # Move to Downloads
                downloads_dir = Path.home() / "Downloads"
                downloads_dir.mkdir(exist_ok=True)
                final_path = downloads_dir / filename
                
                if final_path.exists():
                    final_path.unlink()
                
                shutil.move(str(temp_file), str(final_path))
                temp_dir.rmdir()
                
                installer_path[0] = final_path
                root.after(0, on_download_complete)
            
            except Exception as e:
                root.after(0, lambda: on_download_error(str(e)))
        
        def on_download_complete():
            progress_bar['value'] = 100
            progress_label.config(text="Download concluído!", fg="#27ae60")
            action_btn.config(text="Instalar Agora", command=install_update,
                            bg="#e67e22", activebackground="#d35400")
            close_btn.config(text="Instalar Depois")
        
        def on_download_error(error):
            progress_label.config(text=f"Erro: {error}", fg="#e74c3c")
            action_btn.config(text="Tentar Novamente", command=download_update,
                            bg="#e74c3c", activebackground="#c0392b")
        
        def download_update():
            action_btn.config(state=tk.DISABLED)
            progress_label.config(text="Preparando download...")
            progress_bar['value'] = 0
            progress_bar.pack(fill=tk.X)
            threading.Thread(target=do_download, daemon=True).start()
        
        def install_update():
            if not installer_path[0] or not installer_path[0].exists():
                messagebox.showerror("Erro", "Arquivo do instalador não encontrado")
                return
            
            response = messagebox.askyesno(
                "Instalar Atualização",
                f"O instalador será executado agora.\n\n"
                f"Namsel OCR será atualizado para v{version}.\n\n"
                f"Continuar?",
                icon=messagebox.QUESTION
            )
            
            if response:
                try:
                    if sys.platform.startswith("win"):
                        # ShellExecute the .exe (native launch — no subprocess, no shell,
                        # and installer elevation manifests are honored like a double-click)
                        os.startfile(str(installer_path[0]))
                    elif sys.platform == "darwin":
                        subprocess.Popen(["open", str(installer_path[0])])
                    
                    messagebox.showinfo(
                        "Instalador Executado",
                        "O instalador foi iniciado.\n\n"
                        "Siga as instruções na tela para concluir a atualização."
                    )
                    root.destroy()
                except Exception as e:
                    messagebox.showerror("Erro", f"Falha ao executar o instalador:\n{e}")
        
        action_btn.config(command=download_update)
        root.mainloop()
        
    except Exception:
        pass  # Silently fail if GUI can't be shown


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

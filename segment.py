# encoding: utf-8
try:
    from .classify import load_cls, label_chars
    from .fast_utils import fnormalize, ftrim, gausslogprob, fadd_padding
    from .feature_extraction import normalize_and_extract_features
    from .config_manager import default_config
except ImportError:
    from classify import load_cls, label_chars
    from fast_utils import fnormalize, ftrim, gausslogprob, fadd_padding
    from feature_extraction import normalize_and_extract_features
    from config_manager import default_config
import cv2 as cv
from numpy import floor, uint8, ones, argmax, hstack, mean, std, \
    ceil, load
import numpy as np
from random import gauss
import os
try:
    from .page_elements2 import PageElements
    from .transitions import horizontal_transitions
    from .utils import local_file, check_for_overlap
    from .viterbi_cython import viterbi_cython
except ImportError:
    from page_elements2 import PageElements
    from transitions import horizontal_transitions
    from utils import local_file, check_for_overlap
    from viterbi_cython import viterbi_cython


def _dbg(msg):
    """Print a debug message when debug_output is enabled (branch lives here
    so hot-path call sites stay decision-free)."""
    if default_config.get('debug_output', False):
        print(msg)


def _ink_projection(letter):
    # Ensure we're counting ink pixels (handle both conventions)
    if letter.max() <= 1:
        ink = (1 - letter).astype(np.float32)  # invert: ink pixels become 1
    else:
        ink = (letter < 128).astype(np.float32)
    # Vertical projection: sum ink pixels per column
    return np.sum(ink, axis=0)


def _valley_candidates(projection, margin, threshold):
    # Find local minima below threshold in search region
    candidates = []
    for x in range(margin, len(projection) - margin):
        if projection[x] <= threshold:
            if projection[x] <= projection[x - 1] and projection[x] <= projection[x + 1]:
                candidates.append((x, float(projection[x])))
    return candidates


def _merge_valley_candidates(candidates, merge_dist):
    # Merge nearby candidates (keep the one with lowest ink)
    merged = []
    current_group = [candidates[0]]

    for i in range(1, len(candidates)):
        if candidates[i][0] - current_group[-1][0] <= merge_dist:
            current_group.append(candidates[i])
        else:
            best = min(current_group, key=lambda c: c[1])
            merged.append(best[0])
            current_group = [candidates[i]]

    best = min(current_group, key=lambda c: c[1])
    merged.append(best[0])

    return merged


def find_projection_split_points(letter, char_mean):
    """Find split points in a wide character blob using vertical ink projection.

    Counts ink pixels per column and finds valleys (low-ink columns) that
    indicate boundaries between connected characters.

    Args:
        letter: Binary image array (ink=1, background=0 or ink=0, background=1)
        char_mean: Mean character width for the current line

    Returns:
        List of x-coordinates where the blob should be split, or empty list.
    """
    h, w = letter.shape
    if w < 2 * char_mean * 0.3:
        return []

    projection = _ink_projection(letter)

    if projection.mean() == 0:
        return []

    # Search the middle region (skip 15% edges to avoid cutting into characters)
    margin = max(3, int(w * 0.15))

    threshold = projection.mean() * 0.3

    candidates = _valley_candidates(projection, margin, threshold)

    if not candidates:
        return []

    merge_dist = max(3, int(char_mean * 0.15))
    return _merge_valley_candidates(candidates, merge_dist)


# --- CNN predictor with sklearn fallback ---
# Locate namsel_BUDA_OCR whether the engine sits at the repo root (flat layout:
# ./namsel_BUDA_OCR) or inside a namsel_ocr/ subpackage (../namsel_BUDA_OCR).
_here = os.path.dirname(os.path.abspath(__file__))
_CNN_DIR = next((d for d in (os.path.join(_here, 'namsel_BUDA_OCR'),
                             os.path.join(_here, '..', 'namsel_BUDA_OCR'))
                 if os.path.isdir(d)), os.path.join(_here, '..', 'namsel_BUDA_OCR'))
_CNN_MODEL = os.path.join(_CNN_DIR, 'best_model.pth')
_CNN_MAPPING = os.path.join(_CNN_DIR, 'label_mapping.json')

use_cnn = False
if os.path.exists(_CNN_MODEL) and os.path.exists(_CNN_MAPPING):
    try:
        from namsel_BUDA_OCR.predict import TibetanCNNPredictor
        predictor = TibetanCNNPredictor(_CNN_MODEL, _CNN_MAPPING)
        use_cnn = True
        print("[segment.py] Using CNN predictor")
    except Exception as e:
        print(f"[segment.py] CNN load failed ({e}), falling back to sklearn")
        cls = load_cls('logistic-cls')
else:
    cls = load_cls('logistic-cls')

## commonly called functions
GaussianBlur = cv.GaussianBlur
if use_cnn:
    predict_log_proba = predictor.predict_log_proba
else:
    predict_log_proba = cls.predict_log_proba
boundingRect = cv.boundingRect
char_gaussians = PageElements.char_gaussians


def _normalize_to_32x32(arr):
    """Normalize image to 32x32 for CNN input."""
    buf = np.ones((32, 32), dtype=np.uint8)
    fnormalize(arr, buf)
    return buf

trans_p = load(open(local_file('stack_bigram_logprob32.npz'),'rb'))
trans_p = trans_p[trans_p.files[0]].transpose()
start_p = load(open(local_file('stack_start_logprob32.npz'),'rb'))
start_p = start_p[start_p.files[0]]
n_states = trans_p.shape[0]


def combine_many_boxes(bxs):
    '''Return the largest bounding box using max height and width from all boxes

    bxs is a list of boxes

    returns (x,y,w,h) of the new box
    '''

    if not bxs:
        raise ValueError('No boxes specified')

    new_top = min([b[1] for b in bxs])
    new_bottom = max([b[1]+b[3] for b in bxs])
    new_left = min([b[0] for b in bxs])
    new_right = max([b[0]+b[2] for b in bxs])
    return (new_left, new_top, new_right-new_left, new_bottom-new_top)

def box_attrs(b):
    led = b[0] # left edge
    red = b[0] + b[2] # right edge
    top = b[1]
    bottom = b[1] + b[3]
    return led, red, top, bottom, b[2], b[3]

#@profile
def image_to_32x16_vector(im):
    a = ones((32,16), dtype=uint8)
    h,w = im.shape
    s = min(32.0/h, 16.0/w)
    b = cv.resize(im, (0,0), fy=s, fx=s, interpolation=cv.INTER_AREA)
    a[0:b.shape[0], 0:b.shape[1]] = b
    return a.flatten()


def normalized_scale(arr):
    crss_denom = float(max(horizontal_transitions(arr)))
    if crss_denom > 0:
        crossings_val = (arr.shape[1])/crss_denom
        scale =  35.0 / crossings_val
    else:
        scale = 1.0

    return scale

class CombineBoxesForPage(object):
    def __init__(self, line_info):
        self.widths = []
        self.final_boxes = []
        self.final_indices = []
        self.line_width_means = []
        self.larger_font_lines = []
        self.line_info = line_info
        self.transitions = []
        self.hangoff = line_info.shapes.conf['combine_hangoff']

        for i in range(self.line_info.k):
            initial_boxes = len(line_info.lines_chars[i]) if i < len(line_info.lines_chars) else 0
            initial_chars = line_info.lines_chars[i] if i < len(line_info.lines_chars) else []
            if default_config.get('debug_output', False):
                print(f"[DEBUG] Line {i}: Starting with {initial_boxes} boxes before combination: {initial_chars}")
            self.combine_for_line_ind(self.line_info, lineind=i)
            final_boxes = len(self.final_boxes[i]) if i < len(self.final_boxes) else 0
            if default_config.get('debug_output', False):
                print(f"[DEBUG] Line {i}: {final_boxes} boxes after combination")


        ###########IMPORTANT USE THIS NORMALLY


        self.char_mean = mean(self.widths)
        self.char_std = std(self.widths)
        self.tsek_mean = line_info.shapes.tsek_mean
        self.tsek_std = line_info.shapes.tsek_std
        

    ###########################################
    # (Dead helpers removed 2026-07-02: _low_ink_sort and li_combine_for_line_ind
    # had no callers anywhere in the repo — only combine_for_line_ind is used.)

    @staticmethod
    def _line_extent(line_info, line):
        top = 1000000 # arbitrary high number
        bottom = 0
        for k in line:
            j = line_info.get_box(k)
            if j[1] < top:
                top = j[1]
            if j[1] + j[3] > bottom:
                bottom = j[1] + j[3]
        return top, bottom

    @staticmethod
    def _sum_whitespace(line_info, line):
        whitespace = np.zeros(len(line), dtype=int)
        for p, c in enumerate(line):
            if p + 1 < len(line):
                ab = line_info.get_box(c)
                nab = line_info.get_box(line[p+1])
                ws_diff = nab[0] - (ab[0]+ab[2])
                whitespace[p] = ws_diff
        return whitespace.sum()

    @staticmethod
    def _blank_small_ccs(line_info, lineind, ln_arr, top, firstbox):
        #             Remove small contours when calculating scale
        try:
            for inx in line_info.small_cc_lines_chars[lineind]:
                box = line_info.get_box(inx)
                ln_arr[box[1] - top:box[1]+box[3]-top, box[0]-firstbox[0]:box[0]+box[2]-firstbox[0]] = 1
        except Exception as e:
            _dbg(f"[segment] small-cc blanking skipped for line {lineind}: {e}")

    def _line_scale(self, line_info, lineind, line):
        """Per-line scale factor from horizontal ink transitions (line non-empty)."""
        shapes = line_info.shapes
        top, bottom = self._line_extent(line_info, line)

        firstbox = line_info.get_box(line[0])
        lastbox = line_info.get_box(line[-1])
        sum_whitespace = self._sum_whitespace(line_info, line)

        ln_arr = shapes.img_arr[top:bottom,firstbox[0]:lastbox[0]+lastbox[2]].copy()

        self._blank_small_ccs(line_info, lineind, ln_arr, top, firstbox)

        # Convert to uint8 for horizontal_transitions compatibility
        ln_arr_uint8 = (ln_arr * 255).astype(np.uint8) if ln_arr.dtype != np.uint8 else ln_arr
        crss_denom = float(max(horizontal_transitions(ln_arr_uint8)))
        if crss_denom > 0:
            crossings_val = (ln_arr.shape[1]-sum_whitespace )/crss_denom
            scale =  35.0 / crossings_val
        else:
            scale = 1.0
        return scale

    @staticmethod
    def _log_combine_flags(line_info, i, next_attrs, raw_tsheg, small_tsheg, is_tsheg, tsheg_separation):
        ledn, ren, topn, bottomn, wn, hn = next_attrs
        if i == 34:
            _dbg(f"[SEGMENT DEBUG] Character 34: is_tsheg={raw_tsheg}, is_small_contour_tsheg={small_tsheg}")
        if is_tsheg:
            _dbg(f"[TSHEG DEBUG] Character {i} detected as tsheg: width={wn}, height={hn}, tsek_mean={line_info.shapes.tsek_mean}, height_threshold={1.5 * line_info.shapes.tsek_mean:.1f}")
        if tsheg_separation:
            _dbg(f"[TSHEG SEPARATION] Separating tsheg at index {i}, box=({ledn},{topn},{wn},{hn})")

    @staticmethod
    def _combine_flags(line_info, i, cur_attrs, next_attrs):
        led, red, top, bottom, w, h = cur_attrs
        ledn, ren, topn, bottomn, wn, hn = next_attrs

        # If left edge of next box doesn't overlap cur box
        # separate as 2 different chars
        if not isinstance(i, str):
            is_interior = line_info.shapes.hierarchy[0][i][0] < 0 and line_info.shapes.hierarchy[0][i][1] < 0 and line_info.shapes.hierarchy[0][i][2] < 0 # i.e. it has no peers at its place in the tree... and no children
        else:
            is_interior = False # Its a string, meaning it is the result of a horizontal cut and likely not an interior

        # Check if characters should be separated (gap between them OR current character is a tsheg)
        raw_tsheg = in_tsek_pop(line_info.shapes, wn, topn, top, bottomn, bottom, i)

        # Enhanced tsheg detection: also check if character is in small_contour_indices
        small_tsheg = hasattr(line_info.shapes, 'small_contour_indices') and i in line_info.shapes.small_contour_indices

        # Combine both detection methods
        is_tsheg = raw_tsheg or small_tsheg

        tsheg_separation = (not is_interior and is_tsheg and not hn > 1.5*line_info.shapes.tsek_mean)

        CombineBoxesForPage._log_combine_flags(line_info, i, next_attrs, raw_tsheg, small_tsheg, is_tsheg, tsheg_separation)

        return is_interior, is_tsheg, tsheg_separation

    def _should_separate(self, is_interior, is_tsheg, cur_attrs, next_attrs, i):
        """The original if/elif absorb-or-separate chain, evaluated only when the
        next box overlaps the current group (ledn <= red) and no tsheg_separation."""
        led, red, top, bottom, w, h = cur_attrs
        ledn, ren, topn, bottomn, wn, hn = next_attrs
        # TSHEG PROTECTION: Check if current character is a tsheg before combining
        # If it's a tsheg, treat it as separate even if spatially close
        if is_tsheg:
            _dbg(f"[TSHEG PROTECTION] Preventing combination of tsheg at index {i}, treating as separate")
            return True
        # one box is completely enveloped by the other
        if is_interior:
            return False
        if ((float(min(wn, w)) - abs((red - ledn))) / float(min(wn, w))) < self.hangoff: # amount hanging off end is 30 %, but protect tshegs
            _dbg(f"[TSHEG PROTECTION] Hangoff condition allows combining for character {i} (not a tsheg)")
            return False
        # The overlap is incidental / boxes are not related
        _dbg(f"[CHAR SEPARATION] Incidental overlap, separating character {i}")
        return True

    def _flush_group(self, line_info, fli, flb, cur_ind, scale):
        fli.append(cur_ind)
        x,y,w,h = combine_many_boxes([line_info.get_box(j) for j in cur_ind])
        flb.append((x,y,w,h))
        self.widths.append(w*scale)
        return w

    def _combine_line_boxes(self, line_info, line, scale):
        fli = [] # Final Line indices
        flb = []
        line_widths = []

        line = iter(line)

        # Initialize the current box and its attrs#         BREAKWIDTH = 3.0
        cur_ind = [next(line)]

        # cb is current box, b is the next box
        cb = line_info.get_box(cur_ind[0])

        cur_attrs = box_attrs(cb)

        # Loop through box, combine and close along the way
        for i in line:
            b = line_info.get_box(i)
            next_attrs = box_attrs(b)

            is_interior, is_tsheg, tsheg_separation = self._combine_flags(line_info, i, cur_attrs, next_attrs)

            # Separate when the next box starts past the group's right edge, on
            # tsheg_separation, or when the overlap chain says so; absorb otherwise.
            if next_attrs[0] > cur_attrs[1] or tsheg_separation or self._should_separate(is_interior, is_tsheg, cur_attrs, next_attrs, i):
                self._flush_group(line_info, fli, flb, cur_ind, scale)
                cur_ind = [i]
                cb = b
                cur_attrs = box_attrs(cb)
            else: # There is overlap: absorb into the current group
                cur_ind.append(i)
                bxs = [line_info.get_box(j) for j in cur_ind]
                bxs.append(b)
                cb = combine_many_boxes(bxs)
                cur_attrs = box_attrs(cb)

        w = self._flush_group(line_info, fli, flb, cur_ind, scale)
        line_widths.append(w)
        return fli, flb, line_widths

    @staticmethod
    def _is_char_tsheg(line_info, char_idx, topn, bottomn):
        char_box = line_info.get_box(char_idx)
        by_width = in_tsek_pop(line_info.shapes, char_box[2], char_box[1], topn,
                               char_box[1]+char_box[3], bottomn, char_idx)
        by_small = hasattr(line_info.shapes, 'small_contour_indices') \
            and char_idx in line_info.shapes.small_contour_indices
        return by_width or by_small

    @staticmethod
    def _group_contains_tsheg(line_info, group, topn, bottomn):
        # Check if any character in this group is a tsheg - if so, don't merge
        for char_idx in group:
            if CombineBoxesForPage._is_char_tsheg(line_info, char_idx, topn, bottomn):
                _dbg(f"[TSHEG PROTECTION] Preventing low-ink re-merging of tsheg at index {char_idx}")
                return True
        return False

    def _low_ink_target(self, line_info, lib, box, group):
        """Index of the low-ink box `group` should re-merge into, or None to keep
        it separate (no enclosing low-ink box, or the group contains a tsheg).

        ###  This attempts to remove noise that doesn't fall into a blurred
        ### low-ink box but does get combined according to normal combination rules
        """
        led, red = box_attrs(box)[:2]
        for p, bx in enumerate(lib):
            ledn, ren, topn, bottomn = box_attrs(bx)[:4]
            if led >= ledn-15 and red <= ren+15:
                if self._group_contains_tsheg(line_info, group, topn, bottomn):
                    return None   # keep separate to preserve the tsheg
                return p
        return None

    def _low_ink_regroup(self, line_info, lineind, fli, flb, scale):
        lib = self.line_info.low_ink_boxes[lineind]
        low_ink_segmentation = {}
        not_intr = []
        for d, box in enumerate(flb):
            p = self._low_ink_target(line_info, lib, box, fli[d])
            if p is None:
                not_intr.append(fli[d])
            else:
                low_ink_segmentation.setdefault(p, []).extend(fli[d])

        all_li_seg = list(low_ink_segmentation.values())
        all_li_seg.extend(not_intr)
        newfli = []
        newflb = []
        for j in all_li_seg:
            newfli.append(j)
            x,y,w,h = combine_many_boxes([line_info.get_box(i) for i in j])
            newflb.append([x,y,w,h])
            self.widths.append(w*scale)

        fliflb = list(zip(newfli, newflb))
        fliflb.sort(key=lambda x: x[1][0])

        fli = [i[0] for i in fliflb]
        flb = [i[1] for i in fliflb]
        return fli, flb

    def combine_for_line_ind(self, line_info, lineind=None):
        line = line_info.lines_chars[lineind]

        line.sort(key=lambda x: line_info.get_box(x)[0])

        if not line:
            return []

        scale = self._line_scale(line_info, lineind, line)
        self.transitions.append(scale)

        fli, flb, line_widths = self._combine_line_boxes(line_info, line, scale)

        if line_info.shapes.low_ink:
            fli, flb = self._low_ink_regroup(line_info, lineind, fli, flb, scale)

        self.final_indices.append(fli)
        self.final_boxes.append(flb)
        self.line_width_means.append(mean(line_widths))

def _tsek_bounds(shapes, scale_factor):
    """(min_width, max_width) tsek bounds. Large (scale > 2) images get tighter
    bounds to avoid false positives; normal/small images get permissive ones."""
    if scale_factor > 2.0:
        return max(2, shapes.tsek_mean - 2*shapes.tsek_std), shapes.tsek_mean + 1*shapes.tsek_std
    return max(1, shapes.tsek_mean - 4*shapes.tsek_std), shapes.tsek_mean + 2*shapes.tsek_std


def _dbg_tsek(cur_ind, width, min_width, max_width, scale_factor, width_check, baseline_check):
    if not default_config.get('debug_output', False):
        return
    b = f"width={width}, bounds=[{min_width:.1f}, {max_width:.1f}], scale={scale_factor:.3f}"
    if width_check and baseline_check:
        print(f"[IN_TSEK_POP] Character {cur_ind}: {b} - DETECTED AS TSHEG")
    elif width_check:
        print(f"[IN_TSEK_POP] Character {cur_ind}: {b} - FAILED baseline check")
    elif baseline_check:
        print(f"[IN_TSEK_POP] Character {cur_ind}: {b} - FAILED width check")
    elif cur_ind == 34:
        print(f"[IN_TSEK_POP] Character {cur_ind}: {b} - NOT DETECTED")


def in_tsek_pop(shapes, width, topn, top, bottomn, baseline, cur_ind):
    '''Determine whether a box is of approx tsek-width'''
    # Scale-adaptive tsheg detection criteria
    if not (hasattr(shapes, 'tsek_mean') and shapes.tsek_mean > 0):
        return False

    scale_factor = getattr(shapes, 'global_scale_factor', 1.0)
    min_width, max_width = _tsek_bounds(shapes, scale_factor)

    baseline_tolerance = 1.0 * shapes.tsek_std if shapes.tsek_std > 0 else 5
    baseline_check = (topn - baseline_tolerance <= baseline <= bottomn + baseline_tolerance)
    width_check = (min_width <= width <= max_width)

    _dbg_tsek(cur_ind, width, min_width, max_width, scale_factor, width_check, baseline_check)

    # Note: no adaptive-bounds fallback — it caused false positives for legitimate
    # characters; small-contour detection in page_elements2.py handles that case.
    return bool(width_check and baseline_check)

class Segmenter(object):
    def __init__(self, line_info, break_resolution = 6, draw_outlines=True):
        self.line_info = line_info
        self.draw_outlines = draw_outlines
        self.break_window_resolution = break_resolution
        self.breakwidth = line_info.shapes.conf['break_width']
        self.cached_features = line_info.shapes.cached_features

        if line_info.shapes.conf['segmenter'] == 'experimental':
            self.construct_vector_set_experimental()
        elif line_info.shapes.conf['segmenter'] == 'stochastic':
            self.construct_vector_set_stochastic()


    def _min_variance_breakwidth(self):
        widths = self.final_box_info.widths
        char_mean = self.final_box_info.char_mean
        char_std = self.final_box_info.char_std
        ws = [1.75, 2.5, 2.75, 3.0, 3.6, 4.0, 8.0]

        new_widths = [[] for i in range(len(ws))]
        for wd in widths:
            for i, w in enumerate(ws):
                if wd >= char_mean + w*char_std :

                    splits = int(floor(float(wd)/(char_mean-char_std)))
                    for u in range(splits):
                        new_widths[i].append(char_mean)
                    else:
                        new_widths[i].append(wd)
                else:
                    new_widths[i].append(wd)

        best_var_arg = np.argmin([np.var(wnews) for wnews in new_widths])
        return ws[best_var_arg]



    def _apply_naro_overlap(self, s, nnbox, line_num, oo_scale_l):
        """If this sub-image overlaps a detected naro (ོ), grow it to include the
        naro contour. Returns the (possibly enlarged) (s, nnbox)."""
        naro = self.line_info.check_naro_overlap(line_num, nnbox)
        if naro is False:
            return s, nnbox
        naro_box = self.line_info.get_box(naro)
        nnbox = combine_many_boxes([nnbox, naro_box])
        ss = cv.resize(s, dsize=(0,0), fx=oo_scale_l, fy=oo_scale_l)
        ss = np.vstack((ones((nnbox[3]-ss.shape[0], ss.shape[1]), dtype=ss.dtype), ss))
        ss = hstack((ss, ones((ss.shape[0], nnbox[2] - ss.shape[1]), dtype=ss.dtype)))
        cv.drawContours(ss, [self.line_info.get_contour(naro)], -1, 0, thickness=-1, offset=(-naro_box[0], -naro_box[1]))
        return ss, nnbox

    @staticmethod
    def _fill_small_contours(s, padding_amount):
        # OpenCV version-agnostic findContours (3.x returns 3-tuple, 4.x a 2-tuple)
        contours_result = cv.findContours(s.copy(), mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)
        ctrs, hier = contours_result[-2:]
        for k, b in enumerate(map(boundingRect, ctrs)):
            if (b[2] < 23 or b[3] < 23) and hier[0][k][3] == 0:
                s[b[1]-1:b[1]+b[3]+1, b[0]-1:b[0]+b[2]+1] = 1
        return s[padding_amount:-padding_amount, padding_amount:-padding_amount]

    def _segment_widths_pass(self, letter, widths, box_xy, oo_scale_l, line_num,
                             cur_mean, cur_std, padding_amount):
        """One random-width partition pass: cut `letter` into len(widths) slices,
        classify each. Returns (vecs, boxes, wdthprobs); stops early (as the
        original `break`) if a slice trims to an empty image."""
        x, y = box_xy
        chars = len(widths)
        prev = 0
        vecs = []
        boxes = []
        wdthprobs = 0
        for i, val in enumerate(widths):
            end = letter.shape[1] if i == chars - 1 else prev+val
            wdthprobs += gausslogprob(cur_mean, cur_std, end-prev)

            s = fadd_padding(letter[:, int(prev):int(end)], padding_amount)
            s = self._fill_small_contours(s, padding_amount)
            s, ofst = ftrim(s, new_offset=True)

            if 0 in s.shape:
                break
            nnbox = [x+(prev + ofst['left'])*oo_scale_l, y + (ofst['top']*oo_scale_l), s.shape[1]*oo_scale_l, s.shape[0]*oo_scale_l]
            if line_num is not None:
                s, nnbox = self._apply_naro_overlap(s, nnbox, line_num, oo_scale_l)
            vecs.append(_normalize_to_32x32(s) if use_cnn else normalize_and_extract_features(s))
            boxes.append(nnbox)
            prev += val
        return vecs, boxes, wdthprobs

    def _sample_multichar(self, chars, letter, box_xy, oo_scale_l, line_num, cur_mean, cur_std):
        """Multi-character break: 15 random-width sampling passes, keep the best
        Viterbi score. Returns (best_prob, best_prd, best_boxes)."""
        letter = cv.dilate(letter.copy(), None, iterations=1)
        padding_amount = 3
        best_prob = -np.inf
        best_prd = None
        best_boxes = None
        for n in range(15):
            widths = [gauss(cur_mean, cur_std) for _ in range(chars)]
            vecs, boxes, wdthprobs = self._segment_widths_pass(
                letter, widths, box_xy, oo_scale_l, line_num, cur_mean, cur_std, padding_amount)
            if not vecs:
                continue
            xn = len(vecs)
            if use_cnn:
                vecs = np.array(vecs)  # (xn, 32, 32)
            else:
                vecs = np.array(vecs).reshape(xn, 346)  # 346 is len(vecs[0])

            probs = predict_log_proba(vecs).astype(np.float32)
            if n % 10 == 0 and n != 0:
                cur_mean = self.final_box_info.char_mean*(.97-(3*n/1000.0))

            prob, prds = viterbi_cython(xn, n_states, start_p, trans_p, probs)
            prob = prob + wdthprobs
            if prob > best_prob:
                best_prob = prob
                best_prd = prds
                best_boxes = boxes
        return best_prob, best_prd, best_boxes

    def _classify_single(self, letter, letter_box, oo_scale_l, cur_mean, cur_std):
        """Single-character case: classify the whole letter as one glyph."""
        best_boxes = [letter_box]
        if use_cnn:
            probs = predict_log_proba(_normalize_to_32x32(letter))
        else:
            features = normalize_and_extract_features(letter, debug_coordinates=letter_box)
            probs = predict_log_proba(features.reshape(1, -1))
        amx = probs[0].argmax()
        try:
            startprob = start_p[amx]
        except IndexError:
            startprob = 1e-10
        best_prob = probs[0][amx] + gausslogprob(cur_mean, cur_std, letter_box[2]/oo_scale_l) + startprob
        return best_prob, [amx], best_boxes

    def _sample_widths_method(self, chars, letter, letter_box, oo_scale_l, line_num=None):
        x, y, w, h = letter_box
        cur_mean = self.final_box_info.char_mean*.97
        cur_std = .295*self.final_box_info.char_std
        _dbg(f"[SEGMENT DEBUG] Breaking {chars} chars: cur_mean={cur_mean:.1f}, cur_std={cur_std:.1f}, box_width={w}")

        if chars > 1:
            best_prob, best_prd, best_boxes = self._sample_multichar(
                chars, letter, (x, y), oo_scale_l, line_num, cur_mean, cur_std)
        else:
            best_prob, best_prd, best_boxes = self._classify_single(
                letter, letter_box, oo_scale_l, cur_mean, cur_std)

        final_prob = best_prob
        res = []
        # best_prd is None only when no multichar sampling pass beat -inf (all passes
        # produced empty vecs) — degrade to no boxes rather than crashing (the original
        # raised on `enumerate(None)`).
        if best_prd is not None:
            for i, val in enumerate(best_prd):
                best_boxes[i] = [int(np.round(k)) for k in best_boxes[i]]
                best_boxes[i].extend([float(np.exp(final_prob)), label_chars[val]])
                res.append(best_boxes[i])

        return (final_prob, res)


    def _detach_tsek(self):
        # Unimplemented stub (no-op). The single call site self._detach_tsek()
        # passes no args, so the signature takes none — the future `letter`-based
        # implementation is sketched in the comments below.
        # 1. check if detach makes sense: i.e. will chopping off end result in
        # something that looks and acts like a tsek, in size and position
        # 2. isolate the tsek-part, create a new bounding box for it
        # update the parent box with new dimensions

        #         tsek_part = letter[:, letter.shape[1]-tsek_mean:]


        pass

    def construct_vector_set_stochastic(self):
        # separate attached tsek
        # note this may note go here exactly, but somewhere in this function
        if self.line_info.shapes.conf.get('detach_tsek'):
            self._detach_tsek()

        final_box_info = CombineBoxesForPage(self.line_info)

        self.final_box_info = final_box_info
        final_boxes = final_box_info.final_boxes

        final_indices = final_box_info.final_indices
        scales = final_box_info.transitions

        self.vectors = [[] for i in range(self.line_info.k)]
        self.new_boxes = [[] for i in range(self.line_info.k)] #


        for l in range(len(final_indices)): # for each line
            self._process_stochastic_line(l, final_boxes, final_indices, final_box_info, scales)

        if not any(self.vectors):
            _dbg('no vectors')
            return
        if self.line_info.shapes.detect_o:
            self._combine_naros_stochastic()

    def _process_stochastic_line(self, l, final_boxes, final_indices, final_box_info, scales):
        try:
            scale_l = scales[l]
            oo_scale_l = 1.0/scale_l
        except:
            print(('ERROR AT ', l, len(scales)))
            raise
        try:
            lb = list(range(len(final_indices[l])))
        except IndexError:
            return

        for i in lb: # for each line box
            self._process_stochastic_box(l, i, final_boxes, final_indices,
                                         final_box_info, scale_l, oo_scale_l, self.breakwidth)

    def _draw_char_letter(self, box, lindices):
        """Rasterize the contours of one line box into a fresh letter array."""
        x, y, w, h = box
        letter = ones((h, w), dtype=uint8)
        for k in lindices:
            if not isinstance(k, str):
                letter = self.line_info.shapes.draw_contour_and_children(k, char_arr=letter, offset=(-x, -y))
            else:
                cv.drawContours(letter, [self.line_info.get_contour(k)], -1, 0, thickness=-1, offset=(-x, -y))
        return letter

    @staticmethod
    def _conservative_breakpoint(final_box_info, breakwidth, letter):
        # Scale-normalized threshold from reference char sizes (not raw measurements),
        # deliberately conservative to avoid over-segmenting legitimate wide letters.
        reference_char_mean = 14.23  # From paragraph2.png reference
        reference_char_std = 3.0     # Typical standard deviation
        scale_factor = getattr(final_box_info, 'global_scale_factor', final_box_info.char_mean / reference_char_mean)
        reference_breakpoint = reference_char_mean + (breakwidth + 1.2) * reference_char_std
        bp = reference_breakpoint * scale_factor
        if default_config.get('debug_output', False):
            print(f"[SEGMENT DEBUG] Character analysis: width={letter.shape[1]:.1f}, conservative_breakpoint={bp:.1f}")
            print(f"[SEGMENT DEBUG] Threshold calc: char_mean={final_box_info.char_mean:.3f}, char_std={final_box_info.char_std:.3f}, scale_factor={scale_factor:.3f}")
        return bp

    def _projection_split_box(self, l, i, letter, box, split_points):
        """Cut a wide blob at the projection split points and push each sub-image
        through the classifier. Returns True (caller skips normal processing)."""
        x, y, w, h = box
        _dbg(f"[PROJECTION SPLIT] Found {len(split_points)} split point(s) at columns {split_points} for {letter.shape[1]}px blob")
        boundaries = [0] + split_points + [letter.shape[1]]
        for seg_idx in range(len(boundaries) - 1):
            seg_left = boundaries[seg_idx]
            seg_right = boundaries[seg_idx + 1]
            sub_letter = letter[:, seg_left:seg_right]

            # Skip empty or trivially small sub-images
            if sub_letter.shape[1] < 3 or np.sum(sub_letter < 0.5) < 5:
                _dbg(f"[PROJECTION SPLIT] Skipping empty sub-image {seg_idx}: {sub_letter.shape}")
                continue

            seg_box = [x + seg_left, y, seg_right - seg_left, h]
            self.new_boxes[l].append(seg_box)
            if use_cnn:
                vect = _normalize_to_32x32(sub_letter)
            else:
                vect = normalize_and_extract_features(sub_letter, debug_char_idx=i, debug_coordinates=seg_box, debug_line=l)
            self.vectors[l].append(vect)
            _dbg(f"[PROJECTION SPLIT] Added sub-image {seg_idx}: box={seg_box}, shape={sub_letter.shape}")
        return True

    def _auto_segment_box(self, l, i, letter, orig_box, oo_scale_l, chars):
        """Fallback automatic segmentation via _sample_widths_method over a range of
        candidate char counts; appends the best recognition results."""
        all_choices = []
        for c in range(int(chars), 0, -1):
            line_num = l if self.line_info.shapes.detect_o else None
            all_choices.append(self._sample_widths_method(c, letter, orig_box, oo_scale_l, line_num=line_num))
        mx = max(all_choices)
        for v in mx[-1]:
            self.new_boxes[l].append(v)
            self.vectors[l].append(v)
            self.line_info.shapes.img_arr[v[1]:v[1]+v[3], v[0]+v[2]] = 1

    def _normal_vect_large(self, l, i, letter, lindices, len_lindices, box):
        """Normal single-char vector for the large-box fall-through. Keeps the
        original bare `except:` around the feature cache (catches any error)."""
        x, y, w, h = box
        self.new_boxes[l].append([x, y, w, h])
        if use_cnn:
            vect = _normalize_to_32x32(letter)
        elif len_lindices == 1:
            try:
                vect = self.cached_features[lindices[0]]
            except: #FIXME: should really check key used
                vect = normalize_and_extract_features(letter, debug_char_idx=i, debug_coordinates=[x, y, w, h], debug_line=l)
        else:
            vect = normalize_and_extract_features(letter, debug_char_idx=i, debug_coordinates=[x, y, w, h], debug_line=l)
        self.vectors[l].append(vect)

    def _normal_vect_small(self, l, i, letter, lindices, len_lindices, box):
        """Normal single-char vector for the small-box path. Keeps the original
        `except KeyError` (narrower than the large-box variant on purpose)."""
        x, y, w, h = box
        self.new_boxes[l].append([x, y, w, h])
        if use_cnn:
            vect = _normalize_to_32x32(letter)
        elif len_lindices == 1:
            try:
                vect = self.cached_features[lindices[0]]
            except KeyError:
                vect = normalize_and_extract_features(letter, debug_char_idx=i, debug_coordinates=[x, y, w, h], debug_line=l)
        else:
            vect = normalize_and_extract_features(letter, debug_char_idx=i, debug_coordinates=[x, y, w, h], debug_line=l)
        self.vectors[l].append(vect)

    def _process_stochastic_box(self, l, i, final_boxes, final_indices,
                                final_box_info, scale_l, oo_scale_l, breakwidth):
        ## New draw, takes into account tree hierarchy of contours
        x, y, w, h = final_boxes[l][i]
        lindices = final_indices[l][i]
        len_lindices = len(lindices)

        letter = self._draw_char_letter((x, y, w, h), lindices)

        if w*scale_l >= 1 and h*scale_l >= 1:
            letter = cv.resize(letter, dsize=(0,0), fx=scale_l, fy=scale_l)

        conservative_breakpoint = self._conservative_breakpoint(final_box_info, breakwidth, letter)

        if letter.shape[1] < conservative_breakpoint:   # normal-sized box
            self._normal_vect_small(l, i, letter, lindices, len_lindices, (x, y, w, h))
            return

        # Box too large — try to break it.
        sw = w*scale_l
        sh = h*scale_l
        chars = sw // (final_box_info.char_mean - 1.5*final_box_info.char_std)  # floor division
        chars = min(chars, 4)
        _dbg(f"[SEGMENT DEBUG] Large character detected: sw={sw:.1f}, calculated_chars={chars}")

        # Only segment if we're confident it's clearly more than one character.
        if chars > 1.5:
            # Vertical projection split first (works in the daemon, no GUI needed).
            # Uses the ORIGINAL box (w,h before the sw/sh reassignment below).
            split_points = find_projection_split_points(letter, final_box_info.char_mean)
            if split_points and self._projection_split_box(l, i, letter, (x, y, w, h), split_points):
                return   # sub-images already appended; skip normal processing
            _dbg(f"[PROJECTION SPLIT] No clear split points found for {letter.shape[1]}px blob, using automatic segmentation")
            # Fallback: automatic segmentation.
            self._auto_segment_box(l, i, letter, final_boxes[l][i], oo_scale_l, chars)
            w = sw
            h = sh

        # Normal processing (also the fall-through when chars <= 1.5).
        self._normal_vect_large(l, i, letter, lindices, len_lindices, (x, y, w, h))

    def _naro_char_with_o(self, i, k):
        """Predict the char for box (i,k) and append the naro vowel 'ོ'."""
        probs = predict_log_proba(self.vectors[i][k])
        mx = np.argmax(probs)
        prob = probs[0][mx]
        if use_cnn:
            ch = label_chars[int(predictor.classes_[mx])] + 'ོ'
        else:
            ch = label_chars[mx] + 'ོ'
        self.vectors[i][k] = ch
        return prob, ch

    def _merge_naro_into_box(self, i, k, box, box1):
        """Combine naro `box` with char `box1` at (i,k), stamping the 'ོ' vowel
        onto whichever vector representation this box currently holds."""
        try:
            nbox = list(combine_many_boxes([box, box1]))
        except:
            print((nbox, 'slkfjlkfj'))
            raise
        if isinstance(self.vectors[i][k], str):
            self.vectors[i][k] += 'ོ'
            nbox = box1
            nbox[-1] = self.vectors[i][k]
        elif isinstance(self.vectors[i][k], list):
            if not self.vectors[i][k][-1][-1] == 'ོ':
                pchar = self.vectors[i][k][-1] + 'ོ'
                self.vectors[i][k][-1] = pchar
            nbox = self.vectors[i][k]
        else:
            prob, ch = self._naro_char_with_o(i, k)
            nbox.append(prob)
            nbox.append(ch)
        self.new_boxes[i][k] = nbox

    def _combine_naros_stochastic(self):
        """detect_o post-pass: fold each detected naro (ོ) contour into the char
        box it overlaps. (Only runs when conf['detect_o'] is on.)"""
        for i, line in enumerate(self.new_boxes):
            used_boxes = set()
            for n in self.line_info.line_naros[i]:
                if n in used_boxes:
                    continue
                box = self.line_info.get_box(n)
                x,y,w,h = box
                for k, box1 in enumerate(line):
                    assert isinstance(box1, (list, tuple)), 'error - {}-{}-{}'.format(str(box1), i, k)
                    assert isinstance(box, (list, tuple)), box
                    try:
                        overlap = check_for_overlap(box1, box)
                    except:

                        print((i, k, box1, 'BOX problem'))
                    if overlap:
                        used_boxes.add(n)
                        self._merge_naro_into_box(i, k, box, box1)


    @staticmethod
    def _exp_split_segs(letter, chars, z, char_mean_int, vsum):
        """Slice `letter` into `chars` segments at ink-valley breaklines for one
        (chars, z) candidate. Returns the list of segment sub-images."""
        segs = []
        prev_breakline = 0
        for pos in range(int(chars-1)):
            if char_mean_int - z >= 0:
                upper_range = [int(np.round((pos+1)*(char_mean_int-z))), int(np.round((pos+1)*(char_mean_int+z)))]
                vsum_range = vsum[upper_range[0]:upper_range[1]]
                if vsum_range.any():
                    breakline = int(np.round((pos+1)*(char_mean_int-z) + argmax(vsum_range)))
                else:
                    breakline = None
                if breakline:
                    sg = letter[:, prev_breakline:breakline]
                    prev_breakline = breakline
                else:
                    sg = letter[:, int(np.round(pos*(char_mean_int-z))):int(np.round((pos+1)*(char_mean_int-z)))]
                    prev_breakline = int(np.round((pos+1)*(char_mean_int-z)))
                segs.append(sg)
        segs.append(letter[:, int(np.round((chars-1)*(char_mean_int-z))):])
        return segs

    def _exp_trim_segs(self, segs, seg_bxs, x, y, scale_l, final_box_info):
        """Fill sub-tsek noise, trim, and compute the scaled box for each segment.
        Returns (bxs, nsegs)."""
        bxs = []
        nsegs = []
        prev_w = 0
        for zi, ltb in enumerate(seg_bxs):
            seg = segs[zi]
            for b in ltb:
                if b[2] < (final_box_info.tsek_mean + 4*final_box_info.tsek_std) or b[3] < final_box_info.tsek_mean + 4*final_box_info.tsek_std:
                    seg[b[1]-1:b[1]+b[3]+1, b[0]-1:b[0]+b[2]+1] = True
            seg, ofst = ftrim(seg, new_offset=True)
            bx = [x+prev_w+(ofst['left']/scale_l), y + (ofst['top']/scale_l), seg.shape[1]/scale_l, seg.shape[0]/scale_l]
            prev_w += seg.shape[1]/scale_l
            bxs.append(bx)
            nsegs.append(seg)
        return bxs, nsegs

    @staticmethod
    def _exp_seg_boxes(segs):
        # OpenCV version-agnostic (3.x 3-tuple / 4.x 2-tuple)
        seg_ctrs = [cv.findContours(sg.copy(), mode=cv.RETR_CCOMP, method=cv.CHAIN_APPROX_SIMPLE)[-2:] for sg in segs]
        # Explicit loop so the failing `sgc` is in scope for the diagnostic (in a
        # comprehension it wasn't, so the original except raised NameError instead).
        seg_bxs = []
        for sgc in seg_ctrs:
            try:
                seg_bxs.append([cv.boundingRect(k) for k in sgc[0]])
            except Exception:
                print(sgc)
                raise
        return seg_bxs

    def _exp_extract_xt(self, nsegs, bxs, l):
        if use_cnn:
            return [_normalize_to_32x32(sg) for zi, sg in enumerate(nsegs) if 0 not in sg.shape]
        return [normalize_and_extract_features(sg, debug_char_idx=zi, debug_coordinates=bxs[zi], debug_line=l)
                for zi, sg in enumerate(nsegs) if 0 not in sg.shape]

    def _experimental_seg_pass(self, l, letter, chars, z, char_mean_int, vsum, x, y, scale_l, final_box_info):
        """One (chars, z) segmentation candidate. Returns (prob, prds, bxs, xt)."""
        segs = self._exp_split_segs(letter, chars, z, char_mean_int, vsum)
        segs = [fadd_padding(sg, 2) for sg in segs]
        seg_bxs = self._exp_seg_boxes(segs)
        bxs, nsegs = self._exp_trim_segs(segs, seg_bxs, x, y, scale_l, final_box_info)
        xt = self._exp_extract_xt(nsegs, bxs, l)
        prd_probs = predict_log_proba(xt).astype(np.float32)
        prob, prds = viterbi_cython(prd_probs.shape[0], n_states, start_p, trans_p, prd_probs)
        return np.exp(prob), prds, bxs, xt

    def _break_experimental(self, l, letter, chars, char_mean_int, vsum, x, y, scale_l, final_box_info):
        """Search (chars, z) candidates for the best-Viterbi break of a wide box.
        Returns (best_seq, best_box_dim, best_prob)."""
        best_box_dim = []
        best_prob = 0.0
        best_seq = None
        for c in range(int(chars), 1, -1):
            for z in range(0, 21, 2):
                prob, prds, bxs, xt = self._experimental_seg_pass(
                    l, letter, c, z, char_mean_int, vsum, x, y, scale_l, final_box_info)
                if prob > best_prob:
                    best_prob = prob
                    best_seq = prds
                    best_box_dim = bxs
        if not best_box_dim:
            # Fallback: last candidate's result (matches the original's use of the
            # loop-final prob/prds/bxs when nothing beat best_prob).
            best_prob = prob
            best_seq = prds
            best_box_dim = bxs
        return best_seq, best_box_dim, best_prob

    def _append_experimental_result(self, l, best_seq, best_box_dim, best_prob):
        for u in range(len(best_seq)):
            self.vectors[l].append(label_chars[best_seq[u]])
            best_box = best_box_dim[u]
            best_box = [int(np.round(ii)) for ii in best_box]
            best_box.append(best_prob)
            best_box.append(label_chars[best_seq[u]])
            self.new_boxes[l].append(best_box)
            try:
                self.line_info.shapes.img_arr[best_box[1]:best_box[1]+best_box[3], best_box[0]+best_box[2]] = 1
            except:
                pass

    def _experimental_normal_vect(self, l, i, letter, box):
        x, y, w, h = box
        self.new_boxes[l].append([x, y, w, h])
        if use_cnn:
            vect = _normalize_to_32x32(letter)
        else:
            vect = normalize_and_extract_features(letter, debug_char_idx=i, debug_coordinates=[x, y, w, h], debug_line=l)
        self.vectors[l].append(vect)

    def _process_experimental_box(self, l, i, final_boxes, final_indices, scale_l, char_mean_int, final_box_info):
        x, y, w, h = final_boxes[l][i]
        letter = self._draw_char_letter((x, y, w, h), final_indices[l][i])
        letter = cv.resize(letter, dsize=(0,0), fx=scale_l, fy=scale_l)

        breakpoint = final_box_info.char_mean + self.breakwidth*final_box_info.char_std
        if letter.shape[1] < breakpoint:   # not too large
            self._experimental_normal_vect(l, i, letter, (x, y, w, h))
            return

        sw = w*scale_l
        sh = h*scale_l
        vsum = letter.sum(axis=0)
        chars = sw // (final_box_info.char_mean - 1.5*final_box_info.char_std)  # floor division
        if not (10.0 > chars > 1.0):   # don't break spans > 10 chars
            self._experimental_normal_vect(l, i, letter, (x, y, w, h))
            return

        best_seq, best_box_dim, best_prob = self._break_experimental(
            l, letter, chars, char_mean_int, vsum, x, y, scale_l, final_box_info)
        self._append_experimental_result(l, best_seq, best_box_dim, best_prob)

    def _process_experimental_line(self, l, final_boxes, final_indices, final_box_info, scales):
        try:
            scale_l = scales[l]
        except:
            print(('ERROR AT ', l, len(scales)))
            raise
        char_mean_int = floor(final_box_info.char_mean)

        try:
            lb = list(range(len(final_indices[l])))
        except IndexError:
            print('index error')
            return

        for i in lb: # for each line box
            self._process_experimental_box(l, i, final_boxes, final_indices, scale_l, char_mean_int, final_box_info)

    def construct_vector_set_experimental(self):
        final_box_info = CombineBoxesForPage(self.line_info)
        self.final_box_info = final_box_info
        final_boxes = final_box_info.final_boxes
        final_indices = final_box_info.final_indices
        scales = final_box_info.transitions

        self.vectors = [[] for i in range(self.line_info.k)]
        self.new_boxes = [[] for i in range(self.line_info.k)] #

        for l in range(len(final_indices)): # for each line
            self._process_experimental_line(l, final_boxes, final_indices, final_box_info, scales)

        if not any(self.vectors):
            _dbg('no vectors')
            return
        if self.line_info.shapes.detect_o:
            self._combine_naros_experimental()

    def _experimental_naro_char(self, i, k, prob):
        """Stamp the naro vowel onto vector (i,k) — experimental path (has an
        rbfcls fallback when the CNN isn't in use)."""
        probs = predict_log_proba(self.vectors[i][k])
        mx = np.argmax(probs)
        prob = probs[0][mx]
        if use_cnn:
            ch = label_chars[int(predictor.classes_[mx])] + 'ོ'
        else:
            mx = self.line_info.rbfcls.predict(self.vectors[i][k])[0]
            ch = label_chars[mx] + 'ོ'
        self.vectors[i][k] = ch
        return prob, ch

    def _combine_naros_experimental(self):
        for i, l in enumerate(self.new_boxes):
            for n in self.line_info.line_naros[i]:
                box = self.line_info.get_box(n)
                x, y, w, h = box
                r0 = x+w
                for k, b in enumerate(l):
                    # Gap-tolerant overlap for compound syllables (small gaps between
                    # naro and main character allowed, like Brazilian diacritics).
                    gap_size = max(0, max(b[0] - r0, x - (b[0]+b[2])))
                    char_mean = getattr(self.line_info.shapes, 'char_mean', 15.0)
                    gap_tolerance = char_mean * default_config.get('compound_syllable_gap_tolerance', 0.15)
                    if gap_size <= gap_tolerance:
                        overlap_ratio = 1.0
                    else:
                        overlap_ratio = ((b[2] + w) - abs(b[0] - x) - abs((b[0]+b[2]) - r0)) / (2*float(min(w, b[2])))
                    if overlap_ratio <= .8:
                        continue
                    try:
                        nbox = list(combine_many_boxes([box, b]))
                    except:
                        print((nbox[3]))
                        raise
                    if isinstance(self.vectors[i][k], str):
                        self.vectors[i][k] += 'ོ'
                        nbox = b
                        nbox[-1] = self.vectors[i][k]
                    else:
                        prob, ch = self._experimental_naro_char(i, k, None)
                        nbox.append(prob)
                        nbox.append(ch)
                    self.new_boxes[i][k] = nbox



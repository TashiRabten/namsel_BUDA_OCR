#! /usr/bin/python
# encoding: utf-8
'''Primary routines that manage OCR recognition'''
from PIL import Image
from bisect import bisect
try:
    from .safe_model_io import load_model
except ImportError:
    from safe_model_io import load_model
try:
    from .classify import load_cls
    from .config_manager import Config
except ImportError:
    from classify import load_cls
    from config_manager import Config
import codecs
from cv2 import drawContours
import datetime
try:
    from .fast_utils import fadd_padding
    from .feature_extraction import normalize_and_extract_features
    from .line_breaker import LineCut, LineCluster
except ImportError:
    from fast_utils import fadd_padding
    from feature_extraction import normalize_and_extract_features
    from line_breaker import LineCut, LineCluster
import logging
import numpy as np
import os
try:
    from .page_elements2 import PageElements as PE2
    from .root_based_finder import is_non_std, word_parts
    from .segment import Segmenter, combine_many_boxes
except ImportError:
    from page_elements2 import PageElements as PE2
    from root_based_finder import is_non_std, word_parts
    from segment import Segmenter, combine_many_boxes
import sys
try:
    from .termset import syllables
    from .tparser import parse_syllables
    from .utils import local_file
except ImportError:
    from termset import syllables
    from tparser import parse_syllables
    from utils import local_file
# Import Viterbi with fallback for NumPy compatibility issues
try:
    from viterbi_cython import viterbi_cython
    VITERBI_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import viterbi_cython: {e}")
    VITERBI_AVAILABLE = False
    
    def viterbi_cython(n_observations, n_components, log_startprob, log_transmatT, framelogprob):
        """Fallback implementation when Cython Viterbi is not available"""
        # Simple argmax prediction as fallback
        predictions = []
        for i in range(n_observations):
            predictions.append(np.argmax(framelogprob[i]))
        return 0.0, predictions

# from viterbi_search import viterbi_search, word_bigram
import warnings

## Ignore warnings. THis is mostlu in response to incessant sklearn
## warnings about passing in 1d arrays
warnings.filterwarnings("ignore")
try:
    from .config_manager import default_config
except ImportError:
    from config_manager import default_config

# Only print debug info if debug_output is enabled
if default_config.get('debug_output', False):
    print('ignoring all warnings')
###

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
# Always load sklearn classifiers (needed by PE2 and as fallback)
cls = load_cls('logistic-cls')
rbfcls = load_cls('rbf-cls')

if os.path.exists(_CNN_MODEL) and os.path.exists(_CNN_MAPPING):
    try:
        from namsel_BUDA_OCR.predict import TibetanCNNPredictor
        predictor = TibetanCNNPredictor(_CNN_MODEL, _CNN_MAPPING)
        use_cnn = True
        print("[recognize.py] Using CNN predictor")
    except Exception as e:
        print(f"[recognize.py] CNN load failed ({e}), falling back to sklearn")

if use_cnn:
    predict_log_proba = predictor.predict_log_proba
    predict_proba = predictor.predict_proba
else:
    predict_log_proba = cls.predict_log_proba
    predict_proba = cls.predict_proba

# Trained characters are labeled by number. Open the shelve that contains
# the mappings between the Unicode character and its number label.
try:
    # Character maps are our OWN bundled data — load via safe gzip+JSON
    # (safe_model_io.load_model), not shelve/pickle: data-only, cross-platform,
    # no dbm backend. char_to_dig str->int (allchars); dig_to_char int->str
    # (label_chars) — int keys preserved by the pairs encoding.
    char_to_dig = load_model(local_file('allchars.json.gz'))
    dig_to_char = load_model(local_file('label_chars.json.gz'))
except Exception as e:
    print(f"Warning: Could not load character mappings: {e}")
    char_to_dig = {}
    dig_to_char = {}

## Uncomment the line below when enabling viterbi_hidden_tsek
try:
    gram3 = load_model(local_file('3gram_stack_dict.json.gz'))
except Exception as e:
    print(f"Warning: Could not load 3gram dictionary: {e}")
    gram3 = {}

word_parts = set(word_parts)

PCA_TRANS = False

trs_prob = np.load(open(local_file('stack_bigram_mat.npz'),'rb'))
trs_prob = trs_prob[trs_prob.files[0]]

cdmap = load_model(local_file('extended_char_dig.json.gz'))

# HMM data structures
trans_p = np.load(open(local_file('stack_bigram_logprob32.npz'),'rb'))
trans_p = trans_p[trans_p.files[0]].transpose()
start_p = np.load(open(local_file('stack_start_logprob32.npz'),'rb'))
start_p = start_p[start_p.files[0]]

start_p_nonlog = np.exp(start_p)

## Uncomment below for syllable bigram
syllable_bigram = load_model(local_file('syllable_bigram.json.gz'))

def get_trans_prob(stack1, stack2):
    try:
        return trs_prob[cdmap[stack1], cdmap[stack2]]
    except KeyError:
        print('Warning: Transition matrix char-dig map has not been updated with new chars')
        return .25

def _idx_to_char(predicted_idx):
    """Map a predicted class index to its character. CNN indices remap through
    classes_; a non-CNN out-of-bounds index falls back to the nearest lower key."""
    if use_cnn:
        # CNN output indices map to original labels via classes_
        original_label = int(predictor.classes_[predicted_idx])
        return dig_to_char.get(original_label, '�')
    if predicted_idx in dig_to_char:
        return dig_to_char[predicted_idx]
    # Fallback for out-of-bounds predictions: nearest defined key at or below idx.
    valid_keys = [k for k in dig_to_char.keys() if k <= predicted_idx]
    return dig_to_char[max(valid_keys)] if valid_keys else '�'


def prd_prob(feature_vect):
    '''Predict character and probability from feature vector or 32x32 image.

    Parameters:
    -----------
    feature_vect: numpy array (346-dim feature vector for sklearn, or 32x32 image for CNN)

    Returns:
    --------
    tuple: (character_string, probability_float)
    '''
    try:
        probs = predict_proba(feature_vect)[0]
        predicted_idx = np.argmax(probs)
        prob = probs[predicted_idx]
        char = _idx_to_char(predicted_idx)
        if default_config.get('debug_output', False):
            print(f"[DEBUG] prd_prob: pred={predicted_idx}, prob={prob:.4f}, char='{char}'")
        return char, float(prob)
    except Exception as e:
        if default_config.get('debug_output', False): print(f"[DEBUG] prd_prob error: {e}")
        return '�', 0.1


#############################################
### Post-processing functions ###
#############################################

def viterbi(states, start_p, trans_p, emit_prob):
    '''A basic viterbi decoder implementation

    states: a vector or list of states 0 to n
    start_p: a matrix or vector of start probabilities
    trans_p: a matrix of transition probabilities
    emit_prob: an nxT matrix of per-class output probabilities
        where n is the number of states and t is the number
        of transitions
    '''
    V = [{}]
    path = {}
    for y in states:
        V[0][y] = start_p[y] * emit_prob[0][y]
        path[y] = [y]

    # Run Viterbi for t > 0
    for t in range(1,len(emit_prob)):
        V.append({})
        newpath = {}
        for y in states:
            (prob, state) = max([(V[t-1][y0] * trans_p[y0][y] * emit_prob[t][y], y0) for y0 in states])
            V[t][y] = prob
            newpath[y] = path[state] + [y]
        path = newpath
    (prob, state) = max([(V[len(emit_prob) - 1][y], y) for y in states])
    return ''.join(dig_to_char[s] for s in path[state])

def _state_in_bounds(y, max_valid_state, start_p, emit_prob, trans_p):
    """Whether state y is within every array's bounds and has a character mapping."""
    return (y <= max_valid_state and y in dig_to_char and
            y < len(start_p) and y < len(emit_prob[0]) and y < len(trans_p))


def _vht_init(states, start_p, trans_p, emit_prob):
    """Viterbi base cases: keep only states within all array bounds that have a
    character mapping. Returns (V, path, valid_states)."""
    V = [{}]
    path = {}
    max_valid_state = min(len(start_p) - 1, len(trans_p) - 1,
                          (len(emit_prob[0]) - 1 if emit_prob and len(emit_prob) > 0 else 0))
    valid_states = []
    for y in states:
        if _state_in_bounds(y, max_valid_state, start_p, emit_prob, trans_p):
            V[0][y] = start_p[y] * emit_prob[0][y]
            path[y] = [y]
            valid_states.append(y)
    return V, path, valid_states


def _run_without_tsek(im_path, tsek_dig):
    """Length of the trailing run in im_path that contains no tsek."""
    run = 0
    for i in im_path[::-1]:
        if i == tsek_dig:
            break
        run += 1
    return run


def _vht_tsek_step(V, path, states, t, trans_p, tsek_dig):
    """Odd step: consider inserting a tsek between the previous char and the next."""
    prob_states = []
    for y0 in states:
        im_path = path.get(y0)
        if not im_path:
            continue
        if len(im_path) > 1:
            run = _run_without_tsek(im_path, tsek_dig)
            pr3 = gram3.get(path[y0][-2], {}).get(path[y0][-1], {}).get(tsek_dig, .5) * (1 + run*2)
        else:
            pr3 = .75
        prob_states.append((V[t-1][y0]*trans_p[y0][tsek_dig]*pr3, y0))
    prob, state = max(prob_states)
    V[t][tsek_dig] = prob
    path[tsek_dig] = path[state] + [tsek_dig]


def _vht_emit_step(V, path, states, t, trans_p, emit_prob, tsek_dig):
    """Even step: emit the next observed char, choosing the best predecessor (or a
    tsek predecessor). Returns the rebuilt path dict."""
    newpath = {}
    for y in np.argsort(emit_prob[t//2])[-50:]:
        prob_states = []
        for y0 in states:
            im_path = path.get(y0, [])[-4:]
            t_m2 = V[t-2].get(y0)
            if not im_path or not t_m2:
                continue
            prob_states.append((V[t-2][y0]*trans_p[y0][y]*emit_prob[t//2][y], y0))
        if not prob_states:
            continue
        prob, state = max(prob_states)
        tsek_prob = V[t-1][tsek_dig]*trans_p[tsek_dig][y]*emit_prob[t//2][y]
        if tsek_prob > prob:
            prob = tsek_prob
            state = tsek_dig
        V[t][y] = prob
        newpath[y] = path[state] + [y]
    return newpath


def viterbi_hidden_tsek(states, start_p, trans_p, emit_prob):
    '''Given a series of recognized characters, infer
likely positions of missing punctuation

    Parameters
    --------
    states: the possible classes that can be assigned to (integer codes of stacks)
    start_p: pre-computed starting probabilities of Tibetan syllables
    trans_p: pre-computed transition probabilities between Tibetan stacks
    emit_prob: matrix of per-class probability for t steps

    Returns:
    List of possible string candidates with tsek inserted
    '''
    tsek_dig = char_to_dig['་']
    V, path, states = _vht_init(states, start_p, trans_p, emit_prob)
    if not states:
        print("Error: No valid states after bounds checking")
        return [""]

    num_obs = len(emit_prob)
    for t in range(1, num_obs*2-1):
        V.append({})
        if t % 2 == 1:
            _vht_tsek_step(V, path, states, t, trans_p, tsek_dig)
        else:
            path = _vht_emit_step(V, path, states, t, trans_p, emit_prob, tsek_dig)
        if not list(V[t].keys()):
            print(f"Warning: No valid paths at step {t}, returning empty result")
            return [""]

    (prob, state) = max([(V[len(V)-1][y], y) for y in list(V[len(V)-1].keys())])
    return _get_tsek_permutations(''.join(dig_to_char[s] for s in path[state]))

def _apply_tsek_ops(syls, op):
    """Rebuild a syllable string, dropping each tsek whose op bit is '0'."""
    op = list(op[::-1])
    nstr = []
    for i in syls:
        if i == '་' and op.pop() == '0':
            continue
        nstr.append(i)
    return ''.join(nstr)


def _tsek_candidate_valid(nstr):
    """Valid unless it parses to a non-standard syllable outside the known set."""
    for p in parse_syllables(nstr):
        if is_non_std(p) and p not in syllables:
            print(nstr, 'rejected')
            return False
    print(nstr, 'accepted')
    return True


def _get_tsek_permutations(tsr):
    tsek_count = tsr.count('་')
    syls = parse_syllables(tsr, omit_tsek=False)

    if tsek_count > 8:
        print('too many permutations')
        return [tsr]
    if tsek_count == 0:
        print('no tsek')
        return [tsr]

    all_candidates = []
    ops = [['0', '1'] for i in range(tsek_count)]
    for op in _enumrate_full_paths(ops):
        nstr = _apply_tsek_ops(syls, op)
        if _tsek_candidate_valid(nstr):
            all_candidates.append(nstr)
    return all_candidates if all_candidates else [tsr]

def _enumrate_full_paths(tree):
    if len(tree) == 1:
        return tree[0]
    combs = []
    frow = tree[-1]
    srow = tree[-2]

    for s in srow:
        for f in frow:
            combs.append(s+f)
    tree.pop()
    tree.pop()
    tree.append(combs)
    return _enumrate_full_paths(tree)

def bigram_prob(syl_list):
    return np.prod([syllable_bigram.get(syl_list[i], {}).get(syl_list[i+1], 1e-5) \
                    for i in range(len(syl_list) -1 )])

def max_syllable_bigram(choices):
    best_prob = 0.0
    best_s = ''
    for s in choices:
        print(s, 'is a choice')
        if not isinstance(s, list):
            s = parse_syllables(s)
        prob = bigram_prob(s)
        if prob > best_prob:
            best_prob = prob
            best_s = s
    best_s = '་'.join(best_s)
    return best_prob, best_s

def _bigram_states():
    """HMM states that have a character mapping, limited to the classifier range."""
    max_hmm_states = start_p.shape[0]  # 871
    valid_char_states = set(dig_to_char.keys())
    return [s for s in range(min(max_hmm_states, 799)) if s in valid_char_states]


def _bigram_collect_obs(segmentation):
    """Flatten every classifiable vector across all lines into a single obs list."""
    obs = []
    for line in segmentation.vectors:
        for ob in line:
            if hasattr(ob, 'flatten'):
                obs.append(ob.flatten())
    return obs


def _bigram_syllable_results(emit_p, classes, states):
    """Segment the emission stream at tsek/shad boundaries, Viterbi-decoding each
    syllable run. Returns the interleaved [candidates, punct, candidates, …] list."""
    results = []
    syllable = []
    for em in emit_p:
        char = dig_to_char[int(classes[np.argmax(em)])]
        if char in ('་', '།'):
            if syllable:
                results.append(viterbi_hidden_tsek(states, start_p_nonlog, trs_prob, syllable))
                results.append(char)
                syllable = []
        else:
            syllable.append(em)
    if syllable:
        results.append(viterbi_hidden_tsek(states, start_p_nonlog, trs_prob, syllable))
    return results


def hmm_recognize_bigram(segmentation):
    states = _bigram_states()
    obs = _bigram_collect_obs(segmentation)
    if not obs:
        return (0, '')

    emit_p = predict_proba(obs)
    classes = predictor.classes_ if use_cnn else cls.classes_
    results = _bigram_syllable_results(emit_p, classes, states)

    all_paths = _enumrate_full_paths(results)
    prob, results = max_syllable_bigram(all_paths)
    return (prob, results)

#############################################
### Recognizers
#############################################

def _label_to_char(inx):
    """Map a cached label index to its char, falling back to the nearest defined
    key at or below inx (or the replacement char) when unmapped."""
    if inx in dig_to_char:
        return dig_to_char[inx]
    valid_keys = [k for k in dig_to_char.keys() if k <= inx]
    return dig_to_char[max(valid_keys)] if valid_keys else '�'


def _hmm_small_char_info(segmentation, s, cached_features, cached_pred_prob):
    """Classify one small connected component into a position/char dict for the
    HMM combine pass."""
    bx = list(segmentation.line_info.shapes.get_boxes()[s])
    x, y, w, h = bx
    try:
        feature_vect = cached_features[s]   # guard: KeyError → recompute below
        inx, probs = cached_pred_prob[s]
        prob = probs[inx]
        prd = _label_to_char(inx)
    except:
        cnt = segmentation.line_info.shapes.contours[s]
        char_arr = np.ones((h, w), dtype=np.uint8)
        drawContours(char_arr, [cnt], -1, 0, thickness=-1, offset=(-x, -y))
        if use_cnn:
            from namsel_ocr.segment import _normalize_to_32x32
            feature_vect = _normalize_to_32x32(char_arr)
        else:
            feature_vect = normalize_and_extract_features(char_arr)
        prd, prob = prd_prob(feature_vect)
    return {'x': x, 'y': y, 'w': w, 'h': h, 'box': bx, 'pred': prd, 'prob': prob, 'original_index': s}


def _hmm_validate_small(info, scale_factor):
    """Scale-adaptive validity of a small char: tsheg must pass a size/position
    check; shads and other small chars are always valid."""
    if info['pred'] == '་':  # tsheg
        w, h, y = info['w'], info['h'], info['y']
        min_w = max(2, int(2 * scale_factor)); max_w = min(10, int(8 * scale_factor))
        min_h = max(2, int(2 * scale_factor)); max_h = min(20, int(15 * scale_factor))
        min_y = int(14 * scale_factor); max_y = int(220 * scale_factor)
        return (w >= min_w and w <= max_w and h >= min_h and h <= max_h and y >= min_y and y <= max_y)
    return True


def _hmm_build_positions(segmentation, vectors, new_boxes, small_chars, cached_features, cached_pred_prob):
    """Combine main-char vectors and validated small chars into one x-sorted list."""
    scale_factor = getattr(segmentation.line_info.shapes, 'global_scale_factor', 1.0)
    positions = []
    for i, vector in enumerate(vectors):
        if i < len(new_boxes):
            positions.append({'x': new_boxes[i][0], 'type': 'main', 'vector': vector, 'box': new_boxes[i], 'index': i})
    for s in small_chars:
        info = _hmm_small_char_info(segmentation, s, cached_features, cached_pred_prob)
        if _hmm_validate_small(info, scale_factor):
            positions.append({'x': info['x'], 'type': 'small', 'pred': info['pred'], 'box': info['box'],
                              'prob': info['prob'], 'original_index': info['original_index'], 'validated': True})
    positions.sort(key=lambda c: c['x'])   # stable → main precedes small at equal x
    return positions


def _hmm_rebuild(positions):
    """Rebuild (vectors, new_boxes, left_edges) from the x-sorted position list."""
    vectors, new_boxes, left_edges = [], [], []
    for char in positions:
        if char['type'] == 'main':
            vectors.append(char['vector'])
            new_boxes.append(char['box'])
            left_edges.append(char['x'])
        elif char['type'] == 'small' and char.get('validated', False):
            box_with_meta = char['box'] + [char['prob'], char['pred']]
            vectors.append(box_with_meta)
            new_boxes.append(box_with_meta)
            left_edges.append(char['x'])
    return vectors, new_boxes, left_edges


def _hmm_group_vectors(vectors, new_boxes, tmp_result):
    """Split vectors into contiguous runs of main-char vectors, breaking at
    pre-segmented strings/lists. Manual string labels emit directly to tmp_result.
    Returns (allstrs, allinx)."""
    allstrs, curstr, allinx, curinx = [], [], [], []
    for j, v in enumerate(vectors):
        if isinstance(v, (str, list)):
            allstrs.append(curstr)
            allinx.append(curinx)
            if isinstance(v, str):
                box = new_boxes[j] if j < len(new_boxes) else [0, 0, 0, 0]
                tmp_result.append(box + [1.0, v])  # high confidence for manual labels
            curstr, curinx = [], []
        else:
            curstr.append(v)
            curinx.append(j)
    allstrs.append(curstr)
    allinx.append(curinx)
    return allstrs, allinx


def _hmm_decode_group(group, n_states):
    """Viterbi-decode one contiguous run of main-char vectors. Returns (prds, probs)."""
    probs = predict_log_proba(group)
    if len(probs) == 1:
        _classes = predictor.classes_ if use_cnn else cls.classes_
        return [int(_classes[probs[0].argmax()])], probs
    probs = probs.astype(np.float32)
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*np.int.*deprecated.*")
            warnings.filterwarnings("ignore", message=".*numpy.*has no attribute.*int.*")
            _prb, prds = viterbi_cython(len(probs), n_states, start_p, trans_p, probs)
    except (AttributeError, TypeError, DeprecationWarning):
        # NumPy-compat fallback: per-column argmax
        _classes = predictor.classes_ if use_cnn else cls.classes_
        prds = [int(_classes[np.argmax(row)]) for row in probs]
    return prds, probs


def _hmm_write_group(new_boxes, prds, probs, inx_group):
    """Append (prob, char) to each box in a decoded group."""
    for c in range(len(prds)):
        ind = inx_group[c]
        new_boxes[ind].append(np.exp(probs[c].max()))
        new_boxes[ind].append(dig_to_char.get(int(prds[c]), '�'))


def _hmm_emit(new_boxes, tmp_result, tsek_mean):
    """Emit each box, inserting a space where the gap to the next box is large."""
    for ind in range(len(new_boxes)):
        tmp_result.append(new_boxes[ind])
        if ind + 1 < len(new_boxes) and new_boxes[ind+1][0] - (new_boxes[ind][0] + new_boxes[ind][2]) >= 1.5*tsek_mean:
            tmp_result.append([-1, -1, -1, -1, 1.0, ' '])


def _process_hmm_line(segmentation, l, vectors, n_states, tsek_mean, cached_features, cached_pred_prob):
    """Build one line's recognized-char result via the HMM recognizer: combine +
    rebuild the char stream, Viterbi-decode each main-char run, then emit."""
    new_boxes = segmentation.new_boxes[l]
    small_chars = segmentation.line_info.small_cc_lines_chars[l]
    positions = _hmm_build_positions(segmentation, vectors, new_boxes, small_chars,
                                     cached_features, cached_pred_prob)
    vectors, new_boxes, _left_edges = _hmm_rebuild(positions)

    tmp_result = []
    allstrs, allinx = _hmm_group_vectors(vectors, new_boxes, tmp_result)
    for f, group in enumerate(allstrs):
        if not group:
            continue
        prds, probs = _hmm_decode_group(group, n_states)
        _hmm_write_group(new_boxes, prds, probs, allinx[f])
    _hmm_emit(new_boxes, tmp_result, tsek_mean)
    return tmp_result


def recognize_chars_hmm(segmentation, tsek_insert_method='baseline'):
    '''Recognize characters using segmented char data.

    Parameters:
    --------------------
    segmentation: an instance of PechaCharSegmenter or Segmenter

    Returns:
    --------------
    results: list of lists containing [x,y,width, height, prob, unicode], specifying the
    coordinates of the bounding box of stack, it probability, and its unicode
    characters -- on each line of the page
    '''
    n_states = trans_p.shape[0]
    results = []
    tsek_mean = segmentation.final_box_info.tsek_mean
    cached_features = segmentation.line_info.shapes.cached_features
    cached_pred_prob = segmentation.line_info.shapes.cached_pred_prob
    for l, vectors in enumerate(segmentation.vectors):
        if not vectors:
            print('no vectors...')
            continue
        results.append(_process_hmm_line(segmentation, l, vectors, n_states, tsek_mean,
                                         cached_features, cached_pred_prob))
    return results


def _classify_small_char(segmentation, s, x, y, w, h, cached_features, cached_pred_prob):
    """Classify a small connected component: punctuation (tsheg ་ / shad ། recovered
    from small_contour_indices) vs a regular character. Returns (prd, prob)."""
    if s in segmentation.line_info.shapes.small_contour_indices:
        # Detected as punctuation — recover its original tsheg/shad prediction;
        # default to tsheg when unclear or unavailable (backward compat).
        try:
            inx, _ = cached_pred_prob[s]
            original_pred = dig_to_char[inx] if inx in dig_to_char else "?"
            if original_pred == "།":
                return "།", 1.0
            return "་", 1.0
        except:
            return "་", 1.0
    # Regular character — classify normally.
    try:
        feature_vect = cached_features[s]   # guard: KeyError here → recompute below
        inx, probs = cached_pred_prob[s]
        return dig_to_char[inx], probs[inx]
    except:
        cnt = segmentation.line_info.shapes.contours[s]
        char_arr = np.ones((h, w), dtype=np.uint8)
        drawContours(char_arr, [cnt], -1, 0, thickness=-1, offset=(-x, -y))
        if use_cnn:
            from namsel_ocr.segment import _normalize_to_32x32
            feature_vect = _normalize_to_32x32(char_arr)
        else:
            feature_vect = normalize_and_extract_features(char_arr)
        return prd_prob(feature_vect)


def _insert_char_at(vectors, new_boxes, left_edges, pos, prd, prob, bx):
    """Insert a classified char (prd/prob/bx) at pos, keeping vectors / new_boxes /
    left_edges in sync. Box becomes [x, y, w, h, prob, char]."""
    vectors.insert(pos, prd)
    new_boxes.insert(pos, bx)
    new_boxes[pos].append(float(prob))  # Python float, not numpy
    new_boxes[pos].append(str(prd))     # character as string
    left_edges.insert(pos, bx[0])


def _baseline_bounds(new_boxes, insertion_pos, left_items, right_items):
    """Vertical extent + left/right/mid x of the local box neighborhood around the
    insertion point. Returns (top, bottom, left, right, mid, local_span)."""
    top = 1000000  # arbitrary high number
    bottom = 0
    lower = max(insertion_pos - left_items, 0)
    upper = min(len(new_boxes)-1, insertion_pos+right_items)
    left = new_boxes[lower][0]
    right = new_boxes[upper][0] + new_boxes[upper][2]
    mid = new_boxes[insertion_pos][0] + new_boxes[insertion_pos][2] if insertion_pos < len(new_boxes) else right
    for j in new_boxes[lower:upper]:
        if j[1] < top:
            top = j[1]
        if j[1] + j[3] > bottom:
            bottom = j[1] + j[3]
    local_span = bottom - top
    top, bottom, left, right, mid = [int(np.round(ff)) for ff in [top, bottom, left, right, mid]]
    return top, bottom, left, right, mid, local_span


def _local_baselines(img_arr, top, bottom, left, right, mid, local_span):
    """Local baseline y (min-ink row) on the left and right of the insertion point;
    falls back to the band midpoint if the ink projection can't be taken."""
    try:
        left_sum = img_arr[top:bottom, left:mid].sum(axis=1)
        right_sum = img_arr[top:bottom, mid:right].sum(axis=1)
        bl_left = top + left_sum.argmin()
        bl_right = top + right_sum.argmin() if mid != right else bl_left
    except (IndexError, ValueError):
        bl_left = top + local_span // 2
        bl_right = bl_left
    return bl_left, bl_right


def _tsek_is_valid(img_arr, bx, insertion_pos, vectors, top, bottom, left, right, mid, local_span):
    """Whether a tsek at bx sits plausibly on the local baseline / middle band (a
    permissive OR of baseline-hit, middle-band, end-of-line, or small-tsek)."""
    bl_left, bl_right = _local_baselines(img_arr, top, bottom, left, right, mid, local_span)
    char_middle_y = bx[1] + bx[3] // 2
    line_middle_y = top + local_span // 2
    baseline_tolerance = local_span * 0.4  # 40% tolerance around baseline/middle
    baseline_hit = ((bl_left >= bx[1] and bl_left <= bx[1] + bx[3]) or
                    (bl_right >= bx[1] and bl_right <= bx[1] + bx[3]))
    return (baseline_hit or
            abs(char_middle_y - line_middle_y) <= baseline_tolerance or
            insertion_pos == len(vectors) or
            bx[3] <= local_span * 0.3)


def _adjust_tsek_pos(insertion_pos, new_boxes, x, w):
    """A tsek can render just before its indicated position; nudge left one slot
    when it overlaps a much-wider previous box."""
    if insertion_pos <= len(new_boxes):
        prev_box = new_boxes[insertion_pos-1]
        if 0 <= x - prev_box[0] < w and 2*w < prev_box[2]:
            return insertion_pos - 1
    return insertion_pos


def _place_small_char_baseline(img_arr, bx, prd, prob, insertion_pos, left_items, right_items,
                               vectors, new_boxes, left_edges, tsek_widths):
    """Baseline-method placement of one small char: put a valid tsek on the
    baseline, append an invalid one onto the existing box, or insert other
    punctuation when it fits the local vertical band."""
    x, y, w, h = bx[:4]
    top, bottom, left, right, mid, local_span = _baseline_bounds(new_boxes, insertion_pos, left_items, right_items)
    if prd == '་' and local_span > 0:
        if _tsek_is_valid(img_arr, bx, insertion_pos, vectors, top, bottom, left, right, mid, local_span):
            insertion_pos = _adjust_tsek_pos(insertion_pos, new_boxes, x, w)
            _insert_char_at(vectors, new_boxes, left_edges, insertion_pos, prd, prob, bx)
        else:
            new_boxes[insertion_pos].append(float(prob))
            new_boxes[insertion_pos].append(str(prd))
            left_edges.insert(insertion_pos, bx[0])
            tsek_widths.append(bx[2])
    elif (bx[1] >= top - .25*local_span and bx[1] + bx[3] <= bottom + local_span*.25) or (insertion_pos == len(vectors)):
        _insert_char_at(vectors, new_boxes, left_edges, insertion_pos, prd, prob, bx)


def _place_small_char(segmentation, s, vectors, new_boxes, left_edges, tsek_widths,
                      img_arr, tsek_insert_method, cached_features, cached_pred_prob):
    """Classify and insert one small connected-component (tsek/shad/punctuation)
    at its left-edge-sorted position on the line."""
    bx = list(segmentation.line_info.shapes.get_boxes()[s])
    x, y, w, h = bx
    prd, prob = _classify_small_char(segmentation, s, x, y, w, h, cached_features, cached_pred_prob)
    insertion_pos = bisect(left_edges, x)
    # More left neighbors near line end, more right neighbors near line start, to
    # get enough boxes to define the local baseline.
    left_items, right_items = (12, 5) if insertion_pos >= len(new_boxes) else (6, 12)

    if tsek_insert_method != 'baseline':
        _insert_char_at(vectors, new_boxes, left_edges, insertion_pos, prd, prob, bx)
        return
    _place_small_char_baseline(img_arr, bx, prd, prob, insertion_pos, left_items, right_items,
                               vectors, new_boxes, left_edges, tsek_widths)


def _should_insert_space(new_boxes, i, tsek_mean):
    """Scale-adaptive spacing detection between boxes i and i+1 (OR logic — any
    criterion triggers a space): a traditional large gap, a large gap relative to
    char width, or a substantial missing-tsek gap."""
    gap_width = new_boxes[i+1][0] - (new_boxes[i][0] + new_boxes[i][2])
    avg_char_width = (new_boxes[i][2] + new_boxes[i+1][2]) / 2
    scale_factor = max(1.0, avg_char_width / 15.0)  # 15px = standard char width
    min_gap_threshold = max(4, int(6 * scale_factor))
    min_large_gap_threshold = max(6, int(8 * scale_factor))
    if gap_width >= 2*tsek_mean:
        return True
    if gap_width >= 1.5 * avg_char_width and gap_width >= min_large_gap_threshold:
        return True
    if gap_width >= 1.2 * tsek_mean and gap_width >= min_gap_threshold and gap_width >= 0.6 * avg_char_width:
        return True
    return False


def _normalize_processed_box(new_boxes, i):
    """Trim an already-completed box (>6 elements from the tsheg pass) back to a
    clean [x, y, w, h, prob, char]. Leaves an exactly-6 box unchanged."""
    box = new_boxes[i]
    if len(box) == 7:
        new_boxes[i] = box[:4] + [float(box[4]), str(box[6])]
    elif len(box) > 7:
        prob = box[4] if len(box) > 4 else 1.0
        new_boxes[i] = box[:4] + [float(prob), str(box[-1])]


def _classify_box(new_boxes, i, v):
    """Classify an unprocessed box's vector (or use its pre-segmented string) and
    append (prob, char)."""
    if not isinstance(v, str):
        try:
            prd, prob = prd_prob(v)
        except:
            print(f"Error processing vector {i}: {v}")
            prd, prob = '?', 0.5
    else:
        prd = v
        prob = .95
    if len(new_boxes[i]) < 6:
        try:
            new_boxes[i].append(float(prob))
        except:
            new_boxes[i].append(1.0)
        new_boxes[i].append(str(prd))


def _finalize_box(new_boxes, i, v):
    """Ensure new_boxes[i] is a well-formed [x, y, w, h, prob, char] entry. Shared
    by the spacing loop and the last-vector tail (previously duplicated verbatim)."""
    if len(new_boxes[i]) >= 6:
        _normalize_processed_box(new_boxes, i)
    else:
        _classify_box(new_boxes, i, v)

    if len(new_boxes[i]) != 6:
        print(f"[ERROR] Box {i} has wrong format: {new_boxes[i]}")
    if not isinstance(new_boxes[i][-1], str):
        print(f"[ERROR] Box {i} character is not string: {type(new_boxes[i][-1])} = {new_boxes[i][-1]}")


def _insert_emph_markers(segmentation, vectors, new_boxes, left_edges, emph_markers):
    """Insert emphasis markers at their left-edge-sorted positions on the line."""
    for em in emph_markers:
        bx = list(segmentation.line_info.shapes.get_boxes()[em])
        mkinx = segmentation.line_info.shapes.cached_pred_prob[em][0]
        marker = dig_to_char[mkinx]
        marker_prob = segmentation.line_info.shapes.cached_pred_prob[em][1][mkinx]
        bx.append(marker_prob)
        bx.append(marker)
        insertion_pos = bisect(left_edges, bx[0])
        vectors.insert(insertion_pos, marker)
        new_boxes.insert(insertion_pos, bx)
        left_edges.insert(insertion_pos, bx[0])


def _emit_line_chars(vectors, new_boxes, tsek_mean):
    """Walk a line's vectors, emitting each finalized [x,y,w,h,prob,char] box and a
    space where the gap warrants one. Returns the line's tmp_result."""
    tmp_result = []
    for i, v in enumerate(vectors[:-1]):
        insert_space = _should_insert_space(new_boxes, i, tsek_mean)
        _finalize_box(new_boxes, i, v)
        tmp_result.append(new_boxes[i])
        if insert_space:
            tmp_result.append([-1, -1, -1, -1, 1.0, ' '])  # space char as string
    # Last vector (same unified pathway, no trailing space).
    if len(vectors) > 0:
        i = len(vectors) - 1
        _finalize_box(new_boxes, i, vectors[i])
        tmp_result.append(new_boxes[i])
    return tmp_result


def _process_probout_line(segmentation, l, vectors, tsek_mean,
                          cached_features, cached_pred_prob, tsek_insert_method):
    """Build one line's recognized-char result: insert small chars (tsek/shad) and
    emphasis markers into the box stream, then emit chars + spaces."""
    new_boxes = segmentation.new_boxes[l]
    small_chars = segmentation.line_info.small_cc_lines_chars[l]
    # Line Cut has no emph_lines object, so work around it for now.
    emph_markers = getattr(segmentation.line_info, 'emph_lines', [])
    if emph_markers:
        emph_markers = emph_markers[l]
    img_arr = segmentation.line_info.shapes.img_arr
    left_edges = [b[0] for b in new_boxes]
    tsek_widths = []

    # Consider small chars from the end of the line backward (useful for misplaced
    # tseks, and maybe for TOC — though that should be checked).
    for s in small_chars[::-1]:
        _place_small_char(segmentation, s, vectors, new_boxes, left_edges, tsek_widths,
                          img_arr, tsek_insert_method, cached_features, cached_pred_prob)

    _insert_emph_markers(segmentation, vectors, new_boxes, left_edges, emph_markers)
    return _emit_line_chars(vectors, new_boxes, tsek_mean)


def recognize_chars_probout(segmentation, tsek_insert_method='baseline'):
    '''Recognize characters using segmented char data.

    Parameters:
    --------------------
    segmentation: an instance of PechaCharSegmenter or Segmenter

    Returns:
    --------------
    results: list of lists containing [x,y,width, height, prob, unicode], specifying the
    coordinates of the bounding box of stack, it probability, and its unicode
    characters -- on each line of the page'''
    results = []
    tsek_mean = segmentation.final_box_info.tsek_mean
    cached_features = segmentation.line_info.shapes.cached_features
    cached_pred_prob = segmentation.line_info.shapes.cached_pred_prob

    for l, vectors in enumerate(segmentation.vectors):
        if not vectors:
            print('no vectors...')
            continue
        results.append(_process_probout_line(segmentation, l, vectors, tsek_mean,
                                             cached_features, cached_pred_prob, tsek_insert_method))
    return results

def _vpp_hmm_fix(img_arr, syllable):
    """Re-run the HMM on a non-standard syllable's sub-image. Returns a corrected
    box [x,y,w,h,prob,hmm_res], or None if the fix produced nothing usable."""
    bx = list(combine_many_boxes([ch[0:4] for ch in syllable]))
    arr = img_arr[bx[1]:bx[1]+bx[3], bx[0]:bx[0]+bx[2]]
    arr = fadd_padding(arr, 3)
    try:
        prob, hmm_res = main(arr, Config(line_break_method='line_cut', page_type='book',
                                         postprocess=False, viterbi_postprocess=True, clear_hr=False),
                             page_info={'flname': ''})
    except TypeError:
        print('HMM run exited with an error.')
        prob, hmm_res = 0, ''
    logging.info('VPP Correction: %s\t%s' % (''.join(s[-1] for s in syllable), hmm_res))
    if prob == 0 and hmm_res == '':
        print('hit problem. using unmodified output')
        return None
    bx.append(prob)
    bx.append(hmm_res)
    return bx


def _vpp_flush_syllable(img_arr, syllable, out_line):
    """Append a completed syllable to out_line — either its HMM-corrected box (for
    non-standard syllables) or its original chars unchanged."""
    if not syllable:
        return
    syl_str = ''.join(s[-1] for s in syllable)
    if is_non_std(syl_str) and syl_str not in syllables:
        print(syl_str, 'HAS PROBLEMS. TRYING TO FIX')
        fixed = _vpp_hmm_fix(img_arr, syllable)
        if fixed is not None:
            out_line.append(fixed)
            return
    out_line.extend(syllable)


def viterbi_post_process(img_arr, results):
    '''Go through all results and attempts to correct invalid syllables'''
    final = [[] for i in range(len(results))]
    for i, line in enumerate(results):
        syllable = []
        for j, char in enumerate(line):
            if char[-1] in '་། ' or not word_parts.intersection(char[-1]) or j == len(line)-1:
                _vpp_flush_syllable(img_arr, syllable, final[i])
                final[i].append(char)
                syllable = []
            else:
                syllable.append(char)
        _vpp_flush_syllable(img_arr, syllable, final[i])

    return final

def main(page_array, conf=Config(viterbi_postprocess=False, line_break_method = None, page_type = None), retries=0,
         text=False, page_info={}):
    '''Main procedure for processing a page from start to finish

    Parameters:
    --------------------
    page_array: a 2 dimensional numpy array containing binary pixel data of
        the image

    page_info: dictionary, optional
        A dictionary containing metadata about the page to be recognized.
        Define strings for the keywords "flname" and "volume" if saving
        a serialized copy of the OCR results.

    retries: Used internally when system attempts to reboot a failed attempt

    text: boolean flag. If true, return text rather than char-position data

    Returns:
    --------------
    text: str
        Recognized text for entire page

    if text=False, return character position and label data as a python dictionary
    '''

    print(page_info.get('flname',''))

    confpath = conf.path
    conf = conf.conf

    line_break_method = conf['line_break_method']
    page_type = conf['page_type']

    ### Set the line_break method automatically if it hasn't been
    ### specified beforehand
    if not line_break_method and not page_type:
        if page_array.shape[1] > 2*page_array.shape[0]:
            print('setting page type as pecha')
            line_break_method = 'line_cluster'
            page_type = 'pecha'
        else:
            print('setting page type as book')
            line_break_method = 'line_cut'
            page_type = 'book'

    conf['page_type'] = page_type
    conf['line_break_method'] = line_break_method
    detect_o = conf.get('detect_o', False)
    print('clear hr', conf.get('clear_hr', False))

    results = []
    out = ''
    try:
        ### Get information about the pages
        shapes = PE2(page_array, cls, page_type=page_type,
                     low_ink=conf['low_ink'],
                     flpath=page_info.get('flname',''),
                     detect_o=detect_o,
                     clear_hr =  conf.get('clear_hr', False))
        shapes.conf = conf

        ### Separate the lines on a page
        if page_type == 'pecha':
            k_groups = shapes.num_lines
        shapes.viterbi_post = conf['viterbi_postprocess']

        if line_break_method == 'line_cut':
            line_info = LineCut(shapes)
            if not line_info: # immediately skip to re-run with LineCluster
                sys.exit()
        elif line_break_method == 'line_cluster':
            line_info = LineCluster(shapes, k=k_groups)


        ### Perform segmentation of characters
        segmentation = Segmenter(line_info)

        ###Perform recognition
        if not conf['viterbi_postprocess']:
            # Force use of probout recognizer to fix Line 0 tsheg insertion issues
            # The hmm recognizer has problematic tsheg insertion logic
            if conf['recognizer'] == 'probout':
                results = recognize_chars_probout(segmentation)
            elif conf['recognizer'] == 'hmm':
                results = recognize_chars_probout(segmentation)  # Use probout instead of hmm
            elif conf['recognizer'] == 'kama':
                results = recognize_chars_probout(segmentation)
                results = recognize_chars_kama(results, segmentation)
            if conf['postprocess']:
                results = viterbi_post_process(segmentation.line_info.shapes.img_arr, results)
        else: # Should only be call from *within* a non viterbi run...

            prob, results = hmm_recognize_bigram(segmentation)
            return prob, results


        ### Construct an output string
        output  = []
        for n, line in enumerate(results):
            for m,k in enumerate(line):
#                 if isinstance(k[-1], int):
#                     print n,m,k
#                     page_array[k[1]:k[1]+k[3], k[0]:k[0]+k[2]] = 0
#                     Image.fromarray(page_array*255).show()

                output.append(k[-1])
            output.append('\n')

        out =  ''.join(output)
        print(out)

        if text:
            results = out

        return results
    except:
        ### Retry and assume the error was cause by use of the
        ### wrong line_break_method...
        import traceback;traceback.print_exc()
        if not results and not conf['viterbi_postprocess']:
            print('WARNING', '*'*40)
            print(page_info['flname'], 'failed to return a result.')
            print('WARNING', '*'*40)
            print()
            if line_break_method == 'line_cut' and retries < 1:
                print('retrying with line_cluster instead of line_cut')
                try:
                    return main(page_array, conf=Config(path=confpath, line_break_method='line_cluster', page_type='pecha'), page_info=page_info, retries = 1, text=text)
                except:
                    logging.info('Exited after failure of second run.')
                    return []
        if not conf['viterbi_postprocess']:
            if not results:
                logging.info('***** No OCR output for %s *****' % page_info['flname'])
            return results

def run_main(fl, conf=None, text=False):
    '''Helper function to do recognition'''
    if not conf:
#         conf = Config(low_ink=False, segmenter='stochastic', recognizer='hmm',
#               break_width=2.0, page_type='pecha', line_break_method='line_cluster',
#               line_cluster_pos='center', postprocess=False, detect_o=False,
#               clear_hr = False)
#
        conf = Config(segmenter='stochastic', recognizer='hmm', break_width=2.5,
                      line_break_method='line_cut', postprocess=False,
                      low_ink=False, stop_line_cut=False, clear_hr=True,
                      detect_o=False)

    return main(np.asarray(Image.open(fl).convert('L'))/255, conf=conf,
                page_info={'flname':os.path.basename(fl), 'volume': VOL},
                text=text)


if __name__ == '__main__':
    fls = ['/Users/zach/random-tibetan-tiff.tif']

    lbmethod = 'line_cluster'
    page_type = 'pecha'
    VOL = 'single_volumes'

    def run_main(fl):
        try:
            return main(np.asarray(Image.open(fl).convert('L'))/255,
                        conf=Config(break_width=2.5, recognizer='hmm',
                                    segmenter='stochastic', page_type='pecha',
                                    line_break_method='line_cluster'),
                        page_info={'flname':fl, 'volume': VOL})
        except:
            return []
    import datetime
    start = datetime.datetime.now()
    print('starting')
    outfile = codecs.open('/home/zr/latest-ocr-outfile.txt', 'w', 'utf-8')

    for fl in fls:

        #### line cut
#         ret = main((np.asarray(Image.open(fl).convert('L'))/255),
#            conf=Config(break_width=2., recognizer='probout',
#            segmenter='stochastic', line_break_method='line_cut',
#            postprocess=False, stop_line_cut=False, low_ink=False, clear_hr=True),
#                    page_info={'flname':fl, 'volume': VOL}, text=True)

        #### line cluster
        ret = main((np.asarray(Image.open(fl).convert('L'))/255),
                   conf=Config(segmenter='stochastic', recognizer='hmm',
                               break_width=2.0, page_type='pecha',
                               line_break_method='line_cluster',
                               line_cluster_pos='center', postprocess=False,
                                detect_o=False, low_ink=False, clear_hr=True),
                    page_info={'flname':fl, 'volume': VOL}, text=True)
        outfile.write(ret)
        outfile.write('\n\n')

    print(datetime.datetime.now() - start, 'time taken')
 

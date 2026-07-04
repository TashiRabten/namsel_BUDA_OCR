#! /usr/bin/python
# encoding: utf-8
'''LayoutDetector - Page layout detection and pecha handling, extracted from page_elements2.py'''

import cv2 as cv
import numpy as np
import platform
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import argrelmin
from scipy.interpolate import UnivariateSpline, splrep, splev

try:
    from .config_manager import default_config
    from .utils import invert_bw
    from .fast_utils import to255
except ImportError:
    from config_manager import default_config
    from utils import invert_bw
    from fast_utils import to255

from scipy.ndimage.interpolation import rotate


class LayoutDetector(object):
    """Mixin class providing layout detection methods for PageElements."""

    def get_tops(self):
        return [self.get_boxes()[i][1] for i in self.get_indices()]

    def _line_char_indices(self, content_box_dict):
        """Content-box char indices worth drawing for line detection: wide enough
        vs a tsek (or not spanning the page) and tall enough to be a real char."""
        tsekmeanfloor = np.floor(self.tsek_mean)
        tsekstdfloor = np.floor(self.tsek_std)
        boxes = self.get_boxes()
        return [i for i in content_box_dict['chars']
                if ((boxes[i][2] > (tsekmeanfloor - self.small_coef * tsekstdfloor)
                     or boxes[i][2] < .1 * self.img_arr.shape[1])
                    and boxes[i][3] > 10)]

    def detect_num_lines(self, content_box_dict):
        '''content_box_dict has values {'chars':[], 'b':b, 'boxes':[],
                                'num_boxes':0, 'num_chars':0}

        where chars are the indices of chars in the content box, b is the
        the xywh dimensions of the box, boxes are the sub-boxes of the
        document tree contained in this box (not box chars but large page-
        structuring boxes.

        Note: page_type must be set to "pecha"
        '''

        cbx, cby, cbw, cbh = content_box_dict['b']

        cbox_arr = np.ones((cbh, cbw), dtype=self.img_arr.dtype)

        cv.drawContours(cbox_arr, [self.contours[i] for i in self._line_char_indices(content_box_dict)],
                        -1, 0, thickness=-1, offset=(-cbx, -cby))
        cbox_arr = cbox_arr[5:-5, :] # shorten from the top and bottom to help out trim in the event of small noise
        # Scale-adaptive morphological operations
        scaled_params = self._get_scaled_params()
        erosion_iterations = max(1, int(5 * scaled_params['scale_factor']))
        blur_w = scaled_params['blur_horizontal']
        blur_h = scaled_params['blur_vertical']

        cbox_arr = cv.erode(cbox_arr, None, iterations=erosion_iterations)
        cbox_arr = to255(cbox_arr)
        cv.blur(cbox_arr, (blur_w, blur_h), dst=cbox_arr)

        # Scale-adaptive threshold value
        threshold_value = max(100, int(200 * scaled_params['scale_factor']))
        ret, cbox_arr = cv.threshold(cbox_arr, threshold_value, 1, cv.THRESH_BINARY)

        vsum = cbox_arr.sum(axis=1)

        vsum_smoothed = gaussian_filter1d(vsum, 25) ###DEFAULT
        len_vsum = len(vsum)

        fx = UnivariateSpline(list(range(len_vsum)), vsum_smoothed)
        tck = splrep(list(range(len_vsum)), fx(list(range(len_vsum))))
        y = splev(list(range(len_vsum)), tck, der=1)
        tck = splrep(list(range(len_vsum)), y)
        mins = argrelmin(fx(list(range(len_vsum))))

        ### Filter false peaks that show up from speckles on page
        mins = [m for m in mins[0] if (cbw - vsum[m])/float(cbw) >= .01]

        self.num_lines = len(mins)

    def draw_hough_outline(self, arr):

        arr = invert_bw(arr)
        h = cv.HoughLinesP(arr, 2, np.pi/4, 1, minLineLength=arr.shape[0]*.15, maxLineGap=5) #This
        PI_O4 = np.pi/4
        if h is not None:
            for line in h[0]:
                new = (line[2]-line[0], line[3] - line[1])
                val = (new[0]/np.sqrt(np.dot(new, new)))
                theta = np.arccos(val)
                if theta >= PI_O4: # Vertical line
                    if line[0] < .5*arr.shape[1]:
                        arr[:,:line[0]+12] = 0
                    else:
                        arr[:,line[0]-12:] = 0
                else: # horizontal line
                    if line[2] - line[0] >= .15 * arr.shape[1]:
                        if line[1] < .5 *arr.shape[0]:
                            arr[:line[1]+17, :] = 0
                        else:
                            arr[line[1]-5:,:] = 0

        return ((arr*-1)+1).astype(np.uint8)

    def save_margin_content(self, tree, content_box):
        '''Look at margin content and try to OCR it. Save results in a JSON file
        file of a dictionary object:
        d = {'left':['margin info 1', ...], 'right':['right margin info 1', etc]}

        Margin content is tricky since letters are often not defined as well
        as the main page content. The current OCR implementation also stumbles
        on text with very few characters. Page numbers don't do well for some
        reason...
        '''

        import json
        import os
        content_box_right_edge = tree[content_box]['b'][0] + tree[content_box]['b'][2]
        inset = 20

        right_content = []
        left_content = []
        for brnch in tree:
            if brnch != content_box:
                outer_box = brnch

                if tree[outer_box]['num_chars'] != 0:
                    bx = tree[outer_box]['b']
                    arr = self.img_arr[bx[1]+inset:bx[1]+bx[3]-inset, bx[0]+inset:bx[0]+bx[2]-inset]

                    text = ''
                    if bx[0] > content_box_right_edge:
                        arr = rotate(arr, -90, cval=1)
                        text = construct_page(rec_main(arr, line_break_method='line_cut', page_type='book', page_info={'flname': 'margin content'}))
                        if text:
                            right_content.append(text)
                    else:
                        arr = rotate(arr, 90, cval=1)
                        text = construct_page(rec_main(arr, line_break_method='line_cut', page_type='book', page_info={'flname': 'margin content'}))
                        if text:
                            left_content.append(text)
        outname = os.path.join(os.path.dirname(self.flpath), os.path.basename(self.flpath)[:-4]+'_margin_content.json')
        json.dump({'right': right_content, 'left': left_content}, open(outname, 'w', encoding='utf-8'), ensure_ascii=False)

    def set_pecha_layout(self):
        a = self.img_arr.copy()

        if self.img_arr.shape[1] > 2*self.img_arr.shape[0]:
            self._page_type = 'pecha'
        else:
            self._page_type = 'book'

        if self._page_type == 'pecha': # Page is pecha format
            a = self.draw_hough_outline(a)

        self.img_arr = a.copy()
        self.update_shapes()

        # Ensure proper binarization before contour detection
        if default_config.get('debug_output', False):
            print(f"[CONTOUR_DEBUG] Image shape: {a.shape}, dtype: {a.dtype}, mean: {a.mean():.1f}")
        if a.dtype != np.uint8:
            a = a.astype(np.uint8)

        # Threshold the image to create binary image for contour detection
        if a.mean() > 127:  # White background - invert for black text on white
            _, a_binary = cv.threshold(a, 127, 255, cv.THRESH_BINARY_INV)
            if default_config.get('debug_output', False):
                print("[CONTOUR_DEBUG] Applied THRESH_BINARY_INV")
        else:  # Already inverted or dark background
            _, a_binary = cv.threshold(a, 127, 255, cv.THRESH_BINARY)
            if default_config.get('debug_output', False):
                print("[CONTOUR_DEBUG] Applied THRESH_BINARY")

        if default_config.get('debug_output', False):
            print(f"[CONTOUR_DEBUG] Binary image unique values: {np.unique(a_binary)}")

        # OpenCV version-agnostic approach
        contours_result = cv.findContours(a_binary, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE)
        contours, hierarchy = contours_result[-2:]  # Get last 2 values regardless of OpenCV version

        ## Most of this logic for identifying rectangles comes from the
        ## squares.py sample in opencv source code.
        def angle_cos(p0, p1, p2):
            d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
            return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

        border_boxes = []

        for j,cnt in enumerate(contours):
            cnt_len = cv.arcLength(cnt, True)
            orig_cnt = cnt.copy()
            cnt = cv.approxPolyDP(cnt, 0.02*cnt_len, True)
            if len(cnt) == 4 and cv.contourArea(cnt) > 1000 and cv.isContourConvex(cnt):
                cnt = cnt.reshape(-1, 2)
                max_cos = np.max([angle_cos(cnt[i],
                                            cnt[(i+1) % 4], cnt[(i+2) % 4] )
                                  for i in range(4)])
                if max_cos < 0.1:
                    b = cv.boundingRect(orig_cnt)
                    x,y,w,h = b
                    border_boxes.append(b)

        border_boxes.sort(key=lambda b: (b[0],b[1]))

        def get_edges(b):
            l = b[0]
            r = b[0] + b[2]
            t = b[1]
            b = b[1] + b[3]
            return (l,r,t,b)

        def bid(b):
            return '%d-%d-%d-%d' % (b[0],b[1],b[2],b[3])

        tree = {}
        for b in border_boxes:
            tree[bid(b)] = {'chars':[], 'b':b, 'boxes':[], 'num_boxes':0, 'num_chars':0}

        def b_contains_nb(b,nb):
            l1,r1,t1,b1 = get_edges(b)
            l2,r2,t2,b2 = get_edges(nb)
            return l1 <= l2 and r2 <= r1 and t1 <= t2 and b1 >= b2

        for i, b in enumerate(border_boxes):
            bx,by,bw,bh = b
            self.img_arr[by:by+1,bx+3:bx+bw-3] = 1

            if platform.system() == "Linux":
                self.img_arr[by+bh,by+bh-1:bx+3:bx+bw-3] = 1

            for nb in border_boxes[i+1:]:
                if b_contains_nb(b, nb):
                    tree[bid(b)]['boxes'].append(bid(nb))
                    tree[bid(b)]['num_boxes'] = len(tree[bid(b)]['boxes'])

        self.update_shapes()

        tree_keys = list(tree.keys())
        tree_keys.sort(key=lambda x: tree[x]['num_boxes'])

        ## Assign contours to boxes
        for i in self.get_indices():
            for k in tree_keys:
                box = tree[k]
                b = box['b']

                char_box = self.get_boxes()[i]
                if b_contains_nb(b, char_box):
                    tree[k]['chars'].append(i)
                    tree[k]['num_chars'] = len(tree[k]['chars'])
                    break

        def qualified_box(bx):
            '''Helper function that ignores boxes that contain other boxes.
            This is useful for finding the main content box which should
            be among the innermost boxes that have no box children '''

            if tree[bx]['num_boxes'] == 0:
                return tree[bx]['num_chars']
            else:
                return -1

        content_box = max(tree, key=qualified_box)
        self.indices = [i for i in tree[content_box]['chars'] if self.boxes[i][2] >= 7]

        self.detect_num_lines(tree[content_box])

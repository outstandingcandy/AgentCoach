"""Subset of upstream `utils/utils_heatmap.py` needed by the inference path.

Only ``coords_to_dict`` and ``complete_keypoints`` are ported — the
gaussian-rendering helpers are training-only and the maxpool extractors
live in the project's existing ``pnlcalib/hrnet.py``.
"""

from __future__ import annotations

import copy

import numpy as np
from scipy.stats import linregress


def coords_to_dict(coords, threshold=0.05, ground_plane_only=False):
    kp_list = []
    for batch in range(coords.size()[0]):
        keypoints = {}
        for count, c in enumerate(range(coords.size(1))):
            if coords.size(2) == 1:
                if ground_plane_only and c + 1 in [12, 15, 16, 19]:
                    continue
                if coords[batch, c, 0, -1] > threshold:
                    keypoints[count + 1] = {
                        'x': coords[batch, c, 0, 0].item(),
                        'y': coords[batch, c, 0, 1].item(),
                        'p': coords[batch, c, 0, 2].item(),
                    }
            else:
                if ground_plane_only and c + 1 in [7, 8, 9, 10, 11, 12]:
                    continue
                if coords[batch, c, 0, -1] > threshold and coords[batch, c, 1, -1] > threshold:
                    keypoints[count + 1] = {
                        'x_1': coords[batch, c, 0, 0].item(),
                        'y_1': coords[batch, c, 0, 1].item(),
                        'p_1': coords[batch, c, 0, 2].item(),
                        'x_2': coords[batch, c, 1, 0].item(),
                        'y_2': coords[batch, c, 1, 1].item(),
                        'p_2': coords[batch, c, 1, 2].item(),
                    }
        kp_list.append(keypoints)
    return kp_list


_LINES_LIST = [
    "Big rect. left bottom", "Big rect. left main", "Big rect. left top",
    "Big rect. right bottom", "Big rect. right main", "Big rect. right top",
    "Goal left crossbar", "Goal left post left ", "Goal left post right",
    "Goal right crossbar", "Goal right post left", "Goal right post right",
    "Middle line", "Side line bottom", "Side line left", "Side line right",
    "Side line top", "Small rect. left bottom", "Small rect. left main",
    "Small rect. left top", "Small rect. right bottom", "Small rect. right main",
    "Small rect. right top",
]

_KEYPOINTS_LINE_LIST = [
    ['Side line top', 'Side line left'], ['Side line top', 'Middle line'],
    ['Side line right', 'Side line top'], ['Side line left', 'Big rect. left top'],
    ['Big rect. left top', 'Big rect. left main'], ['Big rect. right top', 'Big rect. right main'],
    ['Side line right', 'Big rect. right top'], ['Side line left', 'Small rect. left top'],
    ['Small rect. left top', 'Small rect. left main'], ['Small rect. right top', 'Small rect. right main'],
    ['Side line right', 'Small rect. right top'], ['Goal left crossbar', 'Goal left post right'],
    ['Side line left', 'Goal left post right'], ['Side line right', 'Goal right post left'],
    ['Goal right crossbar', 'Goal right post left'], ['Goal left crossbar', 'Goal left post left '],
    ['Side line left', 'Goal left post left '], ['Side line right', 'Goal right post right'],
    ['Goal right crossbar', 'Goal right post right'], ['Side line left', 'Small rect. left bottom'],
    ['Small rect. left bottom', 'Small rect. left main'], ['Small rect. right bottom', 'Small rect. right main'],
    ['Side line right', 'Small rect. right bottom'], ['Side line left', 'Big rect. left bottom'],
    ['Big rect. left bottom', 'Big rect. left main'], ['Big rect. right main', 'Big rect. right bottom'],
    ['Side line right', 'Big rect. right bottom'], ['Side line left', 'Side line bottom'],
    ['Side line bottom', 'Middle line'], ['Side line bottom', 'Side line right'],
]

_KEYPOINT_AUX_PAIR_LIST = [
    ['Small rect. left main', 'Side line top'], ['Big rect. left main', 'Side line top'],
    ['Big rect. right main', 'Side line top'], ['Small rect. right main', 'Side line top'],
    ['Small rect. left main', 'Big rect. left top'], ['Big rect. right top', 'Small rect. right main'],
    ['Small rect. left top', 'Big rect. left main'], ['Small rect. right top', 'Big rect. right main'],
    ['Small rect. left bottom', 'Big rect. left main'], ['Small rect. right bottom', 'Big rect. right main'],
    ['Small rect. left main', 'Big rect. left bottom'], ['Small rect. right main', 'Big rect. right bottom'],
    ['Small rect. left main', 'Side line bottom'], ['Big rect. left main', 'Side line bottom'],
    ['Big rect. right main', 'Side line bottom'], ['Small rect. right main', 'Side line bottom'],
]


def complete_keypoints(kp_dict, lines_dict, w, h, normalize=False):
    """Fill missing keypoints by intersecting detected line pairs."""

    def line_intersection(x1, y1, x2, y2):
        x1[-1] += 1e-7
        x2[-1] += 1e-7
        slope1, intercept1, *_ = linregress(x1, y1)
        slope2, intercept2, *_ = linregress(x2, y2)
        x_int = (intercept2 - intercept1) / (slope1 - slope2 + 1e-7)
        y_int = slope1 * x_int + intercept1
        return x_int, y_int

    w_extra = 0.0 * w
    h_extra = 0.0 * h

    complete_dict = copy.deepcopy(kp_dict)
    for key in range(1, 31):
        if key in kp_dict:
            continue
        line_keys = _KEYPOINTS_LINE_LIST[key - 1]
        line_key1 = _LINES_LIST.index(line_keys[0]) + 1
        line_key2 = _LINES_LIST.index(line_keys[1]) + 1
        if all(k in lines_dict for k in [line_key1, line_key2]):
            x1 = [lines_dict[line_key1]['x_1'], lines_dict[line_key1]['x_2']]
            y1 = [lines_dict[line_key1]['y_1'], lines_dict[line_key1]['y_2']]
            x2 = [lines_dict[line_key2]['x_1'], lines_dict[line_key2]['x_2']]
            y2 = [lines_dict[line_key2]['y_1'], lines_dict[line_key2]['y_2']]
            new_kp = line_intersection(x1, y1, x2, y2)
            if -w_extra < new_kp[0] < w_extra + w and -h_extra < new_kp[1] < h_extra + h:
                complete_dict[key] = {'x': round(new_kp[0], 0), 'y': round(new_kp[1], 0), 'p': 1.0}

    for key in range(1, len(_KEYPOINT_AUX_PAIR_LIST)):
        line_keys = _KEYPOINT_AUX_PAIR_LIST[key - 1]
        line_key1 = _LINES_LIST.index(line_keys[0]) + 1
        line_key2 = _LINES_LIST.index(line_keys[1]) + 1
        if all(k in lines_dict for k in [line_key1, line_key2]):
            x1 = [lines_dict[line_key1]['x_1'], lines_dict[line_key1]['x_2']]
            y1 = [lines_dict[line_key1]['y_1'], lines_dict[line_key1]['y_2']]
            x2 = [lines_dict[line_key2]['x_1'], lines_dict[line_key2]['x_2']]
            y2 = [lines_dict[line_key2]['y_1'], lines_dict[line_key2]['y_2']]
            new_kp = line_intersection(x1, y1, x2, y2)
            if -w_extra < new_kp[0] < w_extra + w and -h_extra < new_kp[1] < h_extra + h:
                complete_dict[key + 57] = {'x': round(new_kp[0], 0), 'y': round(new_kp[1], 0), 'p': 1.0}

    if normalize:
        for kp in complete_dict.keys():
            complete_dict[kp]['x'] /= w
            complete_dict[kp]['y'] /= h
        for line in lines_dict.keys():
            lines_dict[line]['x_1'] /= w
            lines_dict[line]['y_1'] /= h
            lines_dict[line]['x_2'] /= w
            lines_dict[line]['y_2'] /= h

    return dict(sorted(complete_dict.items())), lines_dict

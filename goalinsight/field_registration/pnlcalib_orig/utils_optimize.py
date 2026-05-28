"""Subset of upstream `utils/utils_optimize.py` needed by ``utils_calib``.

Drops shapely / matplotlib / optical-flow helpers (those hardcoded
``105/68`` literals and aren't on the calibration path). Function bodies
are otherwise identical to upstream.
"""

from __future__ import annotations

import cv2
import numpy as np


def _is_invertible(a: np.ndarray) -> bool:
    return np.linalg.cond(a) < 1 / np.finfo(a.dtype).eps


def plane_from_P(P, cam_pos, principal_point):
    if not any(np.isnan(P.flatten())):
        H = np.delete(P, 2, axis=1)
        pp = np.array([principal_point[0], principal_point[1], 1.0])
        if _is_invertible(H):
            pp_proj = np.linalg.inv(H) @ pp
        else:
            pp_proj = np.linalg.pinv(H) @ pp
        pp_proj /= pp_proj[-1]
        plane_vector = pp_proj - cam_pos
        return plane_vector, cam_pos
    return None, None


def plane_from_H(H, cam_pos, principal_point):
    if not any(np.isnan(H.flatten())):
        pp = np.array([principal_point[0], principal_point[1], 1.0])
        if _is_invertible(H):
            pp_proj = np.linalg.inv(H) @ pp
        else:
            pp_proj = np.linalg.pinv(H) @ pp
        pp_proj /= pp_proj[-1]
        plane_vector = pp_proj - cam_pos
        return plane_vector, cam_pos
    return None, None


def is_in_front_of_plane(point, plane_normal, plane_point):
    return np.dot(point - plane_point, plane_normal) > 0


def line_plane_intersection(p1, p2, plane_normal, plane_point, epsilon=0.5):
    points_clipped = []
    p1 = np.array(p1)
    p2 = np.array(p2)

    p1_f = is_in_front_of_plane(p1, plane_normal, plane_point)
    p2_f = is_in_front_of_plane(p2, plane_normal, plane_point)
    p_f = [p1_f, p2_f]

    if not p1_f and not p2_f:
        return points_clipped

    if p1_f and p2_f:
        return [p1, p2]

    for count, p in enumerate([p1, p2]):
        if p_f[count]:
            points_clipped.append(p)
        else:
            line_dir = p2 - p1
            denom = np.dot(plane_normal, line_dir)
            if np.isclose(denom, 0):
                continue
            t = np.dot(plane_normal, (plane_point - p1)) / denom
            intersection_point = p1 + t * line_dir
            intersection_point += epsilon * plane_normal / np.linalg.norm(plane_normal)
            points_clipped.append(intersection_point)

    return points_clipped


def get_opt_vector(pos, rot):
    rot_vector, _ = cv2.Rodrigues(rot)
    return np.concatenate((pos, rot_vector.ravel()))


def vector_to_mtx(vector, mtx):
    x_focal_length = mtx[0, 0]
    y_focal_length = mtx[1, 1]
    principal_point = (mtx[0, 2], mtx[1, 2])
    position_meters = vector[:3]
    rot_vector = np.array(vector[3:])
    rotation, _ = cv2.Rodrigues(rot_vector)

    It = np.eye(4)[:-1]
    It[:, -1] = -position_meters
    Q = np.array([
        [x_focal_length, 0, principal_point[0]],
        [0, y_focal_length, principal_point[1]],
        [0, 0, 1],
    ])
    P = Q @ (rotation @ It)
    return P


def point_to_line_distance(l1, l2, p):
    A = (l2[1] - l1[1])
    B = (l2[0] - l1[0])
    C = l2[0] * l1[1] - l2[1] * l1[0]
    num = (A * p[0] - B * p[1] + C)
    den = np.sqrt(A ** 2 + B ** 2)
    if den > 0:
        return num / den
    return 0


from utils.configurations import KEYPOINT_EDGE_INDS_TO_COLOR, COLORS
import cv2


def draw_kps(img, kps, confidence_threshold=0.4):
    for x, y, score in kps:
        if score < confidence_threshold:
            continue
        img = cv2.circle(img, (int(x), int(y)), 4, (255, 0, 0), 2)
        for (e1, e2), color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
            if kps[e1, 2] > confidence_threshold and kps[e2, 2] > confidence_threshold:
                img = cv2.line(img, tuple(kps[e1,:2].astype(int)), tuple(kps[e2,:2].astype(int)), COLORS[color], 2)
    return img

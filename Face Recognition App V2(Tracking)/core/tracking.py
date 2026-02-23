from dataclasses import dataclass
from typing import List, Tuple, Optional
import cv2


@dataclass
class Track:
    label: str
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    tracker: object


def _create_tracker(tracker_type: str):
    t = tracker_type.upper().strip()

    if t == "CSRT":
        if hasattr(cv2, "TrackerCSRT_create"):
            return cv2.TrackerCSRT_create()
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
            return cv2.legacy.TrackerCSRT_create()

    if t == "KCF":
        if hasattr(cv2, "TrackerKCF_create"):
            return cv2.TrackerKCF_create()
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
            return cv2.legacy.TrackerKCF_create()

    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
        return cv2.legacy.TrackerKCF_create()
    if hasattr(cv2, "TrackerKCF_create"):
        return cv2.TrackerKCF_create()

    raise RuntimeError("No supported OpenCV tracker found. Install opencv-contrib-python.")


def init_tracks(frame_bgr, labels: List[str], bboxes_xywh: List[Tuple[int, int, int, int]], tracker_type: str) -> List[Track]:
    tracks: List[Track] = []
    for label, bbox in zip(labels, bboxes_xywh):
        tr = _create_tracker(tracker_type)
        tr.init(frame_bgr, bbox)
        tracks.append(Track(label=label, bbox=bbox, tracker=tr))
    return tracks


def update_tracks(frame_bgr, tracks: List[Track]) -> List[Track]:
    kept: List[Track] = []
    for t in tracks:
        ok, bbox = t.tracker.update(frame_bgr)
        if ok:
            t.bbox = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
            kept.append(t)
    return kept


def _iou_xywh(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    a_x2, a_y2 = ax + aw, ay + ah
    b_x2, b_y2 = bx + bw, by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(a_x2, b_x2)
    inter_y2 = min(a_y2, b_y2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    a_area = aw * ah
    b_area = bw * bh
    union = a_area + b_area - inter_area

    if union <= 0:
        return 0.0
    return inter_area / union


def reconcile_tracks(
    frame_bgr,
    tracks: List[Track],
    det_labels: List[str],
    det_bboxes: List[Tuple[int, int, int, int]],
    tracker_type: str,
    iou_threshold: float = 0.25,
) -> List[Track]:
    if not tracks:
        return init_tracks(frame_bgr, det_labels, det_bboxes, tracker_type)

    used_tracks = set()
    used_dets = set()

    for di, db in enumerate(det_bboxes):
        best_iou = 0.0
        best_ti: Optional[int] = None

        for ti, t in enumerate(tracks):
            if ti in used_tracks:
                continue
            iou = _iou_xywh(t.bbox, db)
            if iou > best_iou:
                best_iou = iou
                best_ti = ti

        if best_ti is not None and best_iou >= iou_threshold:
            used_tracks.add(best_ti)
            used_dets.add(di)

            tracks[best_ti].label = det_labels[di]
            tracks[best_ti].bbox = db

            tr = _create_tracker(tracker_type)
            tr.init(frame_bgr, db)
            tracks[best_ti].tracker = tr

    for di, (lbl, bb) in enumerate(zip(det_labels, det_bboxes)):
        if di in used_dets:
            continue
        tr = _create_tracker(tracker_type)
        tr.init(frame_bgr, bb)
        tracks.append(Track(label=lbl, bbox=bb, tracker=tr))

    return tracks
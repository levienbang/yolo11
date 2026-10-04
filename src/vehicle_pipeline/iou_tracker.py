"""Detector-independent IoU tracker for desktop and TensorRT/Jetson.

Input: [(xyxy_box, class_id, confidence), ...]. NumPy is the only dependency.
Adds motion prediction, global one-to-one assignment, and two confidence passes
to the IoU tracker in jetson_yolo_code/trt_video_track.py.
"""
import numpy as np


def box_iou(a, b):
    """Scalar helper compatible with the original Jetson tracker."""
    return float(_pair_metrics(np.asarray([a]), np.asarray([b]))[0][0, 0])


def _pair_metrics(a, b):
    """Vectorized IoU, normalized center distance and area similarity."""
    wh_a = np.maximum(a[:, 2:] - a[:, :2], 0)
    wh_b = np.maximum(b[:, 2:] - b[:, :2], 0)
    inter_wh = np.maximum(np.minimum(a[:, None, 2:], b[None, :, 2:])
                          - np.maximum(a[:, None, :2], b[None, :, :2]), 0)
    inter = inter_wh[..., 0] * inter_wh[..., 1]
    area_a = wh_a[:, 0] * wh_a[:, 1]
    area_b = wh_b[:, 0] * wh_b[:, 1]
    union = area_a[:, None] + area_b[None, :] - inter
    iou = inter / np.maximum(union, 1e-9)
    centers_a = (a[:, :2] + a[:, 2:]) * 0.5
    centers_b = (b[:, :2] + b[:, 2:]) * 0.5
    distance = np.linalg.norm(centers_a[:, None] - centers_b[None, :], axis=2)
    diagonal = np.minimum(np.linalg.norm(wh_a, axis=1)[:, None],
                          np.linalg.norm(wh_b, axis=1)[None, :])
    distance /= np.maximum(diagonal, 1.0)
    ratio = np.minimum(area_a[:, None], area_b[None, :]) / np.maximum(
        np.maximum(area_a[:, None], area_b[None, :]), 1e-9)
    return iou, distance, ratio


def _assign(cost):
    """Rectangular Hungarian assignment, no SciPy/lap dependency.

Rows <= columns. Dummy columns supplied by the caller allow unmatched rows.
"""
    n, m = cost.shape
    if not n:
        return []
    u, v = np.zeros(n + 1), np.zeros(m + 1)
    p, way = np.zeros(m + 1, dtype=np.int32), np.zeros(m + 1, dtype=np.int32)
    for i in range(1, n + 1):
        p[0], j0 = i, 0
        minimum, used = np.full(m + 1, np.inf), np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            available = np.flatnonzero(~used[1:]) + 1
            reduced = cost[i0 - 1, available - 1] - u[i0] - v[available]
            better = reduced < minimum[available]
            columns = available[better]
            minimum[columns], way[columns] = reduced[better], j0
            j1 = int(available[np.argmin(minimum[available])])
            delta = minimum[j1]
            u[p[used]] += delta
            v[used] -= delta
            minimum[~used] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    return [(int(p[j] - 1), j - 1) for j in range(1, m + 1) if p[j]]


class IoUTracker:
    def __init__(self, iou_threshold=0.25, max_age=30,
                 high_conf=0.20, low_conf=0.05, center_gate=0.5):
        if not (0 <= iou_threshold <= 1 and max_age >= 0
                and 0 <= low_conf <= high_conf <= 1 and center_gate > 0):
            raise ValueError('Invalid tracker thresholds')
        self.iou_threshold, self.max_age = iou_threshold, max_age
        self.high_conf, self.low_conf, self.center_gate = high_conf, low_conf, center_gate
        self.next_id, self.tracks = 1, []

    def _match(self, track_indices, detection_indices, predicted, boxes, classes, scores):
        matched = []
        # Class groups reduce assignment cost and prevent cross-class ID matching.
        for cls in sorted(set(classes[detection_indices].tolist())):
            ti = [i for i in track_indices if self.tracks[i]['cls'] == cls]
            di = [i for i in detection_indices if classes[i] == cls]
            if not ti:
                continue
            iou, distance, ratio = _pair_metrics(predicted[ti], boxes[di])
            mature = np.asarray([self.tracks[i]['hits'] >= 2 for i in ti])[:, None]
            gate = np.asarray([self.center_gate if self.tracks[i]['age'] < 8
                               else self.center_gate * 0.5 for i in ti])[:, None]
            valid = (iou >= self.iou_threshold) | (mature & (distance <= gate) & (ratio >= 0.5))
            affinity = (0.75 * iou + 0.25 * np.maximum(0, 1 - distance / self.center_gate))
            affinity *= 0.9 + 0.1 * scores[di][None, :]
            # Rows are detections, columns are tracks + unmatched choices.
            cost = np.full((len(di), len(ti) + len(di)), 1.0)
            cost[:, :len(ti)] = np.where(valid, 1 - affinity, 1e6).T
            for d, t in _assign(cost):
                if t < len(ti) and valid[t, d]:
                    matched.append((ti[t], di[d]))
        return matched

    def update(self, detections):
        # Preserve original indices: the caller uses them to select the right mask.
        for track in self.tracks:
            track['detection_index'] = None
        clean = []
        for index, (box, cls, score) in enumerate(detections):
            box = np.asarray(box, dtype=np.float32)
            if (box.shape == (4,) and np.all(np.isfinite(box)) and np.isfinite(score)
                    and box[2] > box[0] and box[3] > box[1] and score >= self.low_conf):
                clean.append((index, box, int(cls), float(score)))
        boxes = np.asarray([d[1] for d in clean], dtype=np.float32).reshape(-1, 4)
        classes = np.asarray([d[2] for d in clean], dtype=np.int32)
        scores = np.asarray([d[3] for d in clean], dtype=np.float32)
        predicted = []
        for track in self.tracks:
            offset = track['velocity'] * min(track['age'] + 1, 10)
            predicted.append(track['box'] + np.tile(offset, 2))
        predicted = np.asarray(predicted, dtype=np.float32).reshape(-1, 4)
        strong = np.flatnonzero(scores >= self.high_conf).tolist()
        weak = np.flatnonzero(scores < self.high_conf).tolist()
        matches = self._match(list(range(len(self.tracks))), strong, predicted, boxes, classes, scores)
        used_tracks, used_detections = {t for t, _ in matches}, {d for _, d in matches}
        remaining = [i for i in range(len(self.tracks)) if i not in used_tracks]
        second = self._match(remaining, weak, predicted, boxes, classes, scores)
        matches.extend(second)
        used_tracks.update(t for t, _ in second)
        used_detections.update(d for _, d in second)
        for ti, di in matches:
            track = self.tracks[ti]
            old_center = (track['box'][:2] + track['box'][2:]) * 0.5
            new_center = (boxes[di, :2] + boxes[di, 2:]) * 0.5
            measured = (new_center - old_center) / (track['age'] + 1)
            track['velocity'] = measured if track['hits'] == 1 else 0.7 * track['velocity'] + 0.3 * measured
            track.update(box=boxes[di].copy(), score=float(scores[di]), age=0,
                         hits=track['hits'] + 1, prev_center=old_center,
                         detection_index=clean[di][0])
        for i, track in enumerate(self.tracks):
            if i not in used_tracks:
                track['age'] += 1
        self.tracks = [t for t in self.tracks if t['age'] <= self.max_age]
        # Weak detections can recover existing IDs, but cannot create noisy IDs.
        for di in strong:
            if di not in used_detections:
                self.tracks.append({'id': self.next_id, 'box': boxes[di].copy(),
                                    'cls': int(classes[di]), 'score': float(scores[di]),
                                    'age': 0, 'hits': 1, 'prev_center': None,
                                    'velocity': np.zeros(2, dtype=np.float32),
                                    'detection_index': clean[di][0]})
                self.next_id += 1
        return self.tracks

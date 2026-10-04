"""Mask/box overlap, matching, AP and confusion visualization for Vietnam evaluation."""
import csv
import cv2
import numpy as np
from vehicle_pipeline.vehicle_benchmark import CLASSES

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


def region(mask):
    """Crop a binary mask, retaining its original image coordinates."""
    points = cv2.findNonZero(mask.astype(np.uint8))
    if points is None:
        return (0, 0, np.zeros((0, 0), dtype=bool), 0)
    x, y, w, h = cv2.boundingRect(points)
    crop = mask[y:y+h, x:x+w].astype(bool)
    return (x, y, crop, int(crop.sum()))


def region_iou(a, b):
    ax, ay, am, area_a = a
    bx, by, bm, area_b = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax+am.shape[1], bx+bm.shape[1]), min(ay+am.shape[0], by+bm.shape[0])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = np.logical_and(am[y1-ay:y2-ay, x1-ax:x2-ax],
                                  bm[y1-by:y2-by, x1-bx:x2-bx]).sum()
    union = area_a + area_b - intersection
    return float(intersection / union) if union else 0.0


def box_iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = max(0, a[2]-a[0])*max(0, a[3]-a[1]) + max(0, b[2]-b[0])*max(0, b[3]-b[1]) - intersection
    return float(intersection/union) if union else 0.0


def matching(record, method, minimum_conf, threshold=0.5, class_id=None):
    """Confidence-ordered, class-aware one-to-one matching; duplicates are FP."""
    scores = record['scores']
    pred_classes, gt_classes = record['pred_classes'], record['gt_classes']
    candidates = [i for i in range(len(scores)) if scores[i] >= minimum_conf
                  and (class_id is None or pred_classes[i] == class_id)]
    candidates.sort(key=lambda i: (-scores[i], i))
    matches, used_gt = {}, set()
    for i in candidates:
        valid = [j for j in range(len(gt_classes)) if j not in used_gt and gt_classes[j] == pred_classes[i]]
        if not valid:
            continue
        j = max(valid, key=lambda j: record['ious'][method][i, j])
        if record['ious'][method][i, j] >= threshold - 1e-9:
            matches[i] = j
            used_gt.add(j)
    targets = [j for j in range(len(gt_classes)) if class_id is None or gt_classes[j] == class_id]
    return candidates, matches, [j for j in targets if j not in used_gt]


def average_precision(records, method, class_id, threshold):
    count = sum(sum(c == class_id for c in r['gt_classes']) for r in records)
    if not count:
        return None
    events = []
    for image_index, record in enumerate(records):
        candidates, matches, _ = matching(record, method, 0.0, threshold, class_id)
        events.extend((record['scores'][i], image_index, i, i in matches) for i in candidates)
    events.sort(key=lambda row: (-row[0], row[1], row[2]))
    if not events:
        return 0.0
    true_positive = np.cumsum([row[3] for row in events])
    false_positive = np.cumsum([not row[3] for row in events])
    recall = true_positive / count
    precision = true_positive / np.maximum(true_positive+false_positive, 1)
    # 101 recall samples with monotonically interpolated precision.
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    samples = []
    for target_recall in np.linspace(0, 1, 101):
        indices = np.flatnonzero(recall >= target_recall)
        samples.append(float(precision[indices[0]]) if len(indices) else 0.0)
    return float(np.mean(samples))


def summarize(records, method, conf, class_id=None, include_ap=True):
    tp, fp, fn, overlaps = 0, 0, 0, []
    for record in records:
        candidates, matches, missed = matching(record, method, conf, class_id=class_id)
        tp += len(matches)
        fp += len(candidates)-len(matches)
        fn += len(missed)
        overlaps.extend(float(record['ious'][method][i, j]) for i, j in matches.items())
    precision = tp/(tp+fp) if tp+fp else 0.0
    recall = tp/(tp+fn) if tp+fn else 0.0
    result = {'images': len(records), 'gt': tp+fn, 'tp': tp, 'fp': fp, 'fn': fn,
              'precision': precision, 'recall': recall,
              'f1': 2*precision*recall/(precision+recall) if precision+recall else 0.0,
              'mean_matched_iou': float(np.mean(overlaps)) if overlaps else None}
    if include_ap:
        class_ids = [class_id] if class_id is not None else range(len(CLASSES))
        ap = [[average_precision(records, method, c, t) for t in IOU_THRESHOLDS] for c in class_ids]
        ap = [values for values in ap if values[0] is not None]
        result.update(map50=float(np.mean([v[0] for v in ap])) if ap else None,
                      map50_95=float(np.mean(ap)) if ap else None)
    return result


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def full_mask(cropped, shape):
    x, y, pixels, _ = cropped
    mask = np.zeros(shape, np.uint8)
    mask[y:y+pixels.shape[0], x:x+pixels.shape[1]] = pixels
    return mask


def matrix_image(matrix, path, normalized=False):
    values = matrix.astype(float)
    if normalized:
        values = values / np.maximum(values.sum(axis=1, keepdims=True), 1)
    canvas = np.full((640, 800, 3), 255, dtype=np.uint8)
    labels = CLASSES + ['background']
    cv2.putText(canvas, 'Rows: GT / Columns: prediction', (24, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,0), 2)
    for c, name in enumerate(labels):
        cv2.putText(canvas, name, (185+c*118, 80), cv2.FONT_HERSHEY_SIMPLEX, .42, (0,0,0), 1)
    maximum = max(1., float(values.max())) if not normalized else 1.
    for r, name in enumerate(labels):
        cv2.putText(canvas, name, (10, 150+r*95), cv2.FONT_HERSHEY_SIMPLEX, .55, (0,0,0), 1)
        for c in range(5):
            x, y = 180+c*118, 100+r*95
            strength = values[r,c]/maximum
            color = (255, int(255-150*strength), int(255-200*strength))
            cv2.rectangle(canvas, (x,y), (x+116,y+93), color, -1)
            label = f'{values[r,c]:.1%}' if normalized else str(int(values[r,c]))
            cv2.putText(canvas, label, (x+14,y+54), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,0,0), 1)
    if not cv2.imwrite(str(path), canvas): raise RuntimeError(f'Cannot save {path}')

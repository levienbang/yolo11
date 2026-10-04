"""Four vehicle classes, dataset fingerprints and spatial confusion accounting."""
import hashlib
import cv2
import numpy as np

CLASSES = ['car', 'truck', 'bus', 'motorcycle']


def image_digest(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f'Cannot decode image: {path}')
    return hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()


def confusion(record, conf, threshold):
    """Rows=GT, columns=prediction; final row/column is background.

    Confidence-ordered greedy spatial matching deliberately ignores class.
    """
    matrix = np.zeros((5, 5), dtype=int)
    matches, used = {}, set()
    candidates = sorted((i for i, score in enumerate(record['scores']) if score >= conf),
                        key=lambda i: (-record['scores'][i], i))
    for i in candidates:
        available = [j for j in range(len(record['gt_classes'])) if j not in used]
        j = max(available, key=lambda j: record['ious']['box_detection'][i, j]) if available else None
        if j is not None and record['ious']['box_detection'][i, j] >= threshold:
            used.add(j)
            matches[i] = j
            matrix[record['gt_classes'][j], record['pred_classes'][i]] += 1
        else:
            matrix[4, record['pred_classes'][i]] += 1
    for j, cid in enumerate(record['gt_classes']):
        if j not in used:
            matrix[cid, 4] += 1
    return matrix, matches


def summarize_confusion(matrix):
    rows = {}
    for c, name in enumerate(CLASSES):
        tp = int(matrix[c, c])
        fp, fn = int(matrix[:, c].sum()-tp), int(matrix[c].sum()-tp)
        p, r = tp/(tp+fp) if tp+fp else 0., tp/(tp+fn) if tp+fn else 0.
        rows[name] = dict(tp=tp, fp=fp, fn=fn, gt=tp+fn, precision=p, recall=r,
                          f1=2*p*r/(p+r) if p+r else 0.)
    matched = int(matrix[:4, :4].sum())
    errors = {}
    for a, b in [(0, 1), (1, 0)]:
        total, localized = int(matrix[a].sum()), int(matrix[a, :4].sum())
        errors[f'{CLASSES[a]}_to_{CLASSES[b]}'] = {
            'count': int(matrix[a, b]), 'gt_count': total, 'localized_gt_count': localized,
            'rate_all_gt': int(matrix[a, b])/total if total else None,
            'rate_localized_gt': int(matrix[a, b])/localized if localized else None}
    return dict(per_class=rows, confusion_matrix=matrix.tolist(), confusion=errors,
                matched_classification_accuracy=float(np.trace(matrix[:4, :4]))/matched if matched else None,
                matched_objects=matched)

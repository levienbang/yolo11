"""Desktop YOLO inference with independent Jetson IoU tracking for hull comparison."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter

from .device import resolve_device
from .hull_common import compute_hull, extract_vector_features
from .iou_tracker import IoUTracker

ROOT = Path(__file__).resolve().parents[2]
COLORS = {'car': (255, 255, 0), 'motorcycle': (255, 255, 255),
          'motorbike': (255, 255, 255), 'bus': (0, 0, 255), 'truck': (0, 255, 255)}


def parse_args(default_algorithm, default_model=None, default_output=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=ROOT / 'videos' / 'vh1.mp4')
    p.add_argument('--model', type=Path, default=default_model or ROOT / 'weights' / 'yolo-nano' / 'yolo11n-seg.pt',
                   help='Segmentation checkpoint; default YOLO11n-seg pretrained COCO.')
    p.add_argument('--algorithm', choices=['quickhull', 'ograham', 'both'], default=default_algorithm)
    p.add_argument('--output-dir', type=Path, default=default_output or ROOT / 'outputs' / 'vh1-nano')
    p.add_argument('--device', default='auto', help='auto, cpu, mps or a CUDA index')
    p.add_argument('--conf', type=float, default=0.05)
    p.add_argument('--imgsz', type=int, default=960)
    p.add_argument('--max-det', type=int, default=300)
    p.add_argument('--track-iou', type=float, default=0.25)
    p.add_argument('--track-max-age', type=int, default=30)
    p.add_argument('--max-frames', type=int, default=0)
    p.add_argument('--show', action='store_true')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    if args.max_frames < 0 or not 0 < args.conf <= 1 or args.imgsz < 32 or args.max_det < 1:
        p.error('max-frames >= 0, 0 < conf <= 1, imgsz >= 32, max-det >= 1 required')
    if not 0 <= args.track_iou <= 1 or args.track_max_age < 0:
        p.error('0 <= track-iou <= 1, track-max-age >= 0 required')
    return args


def run(args):
    import cv2
    import numpy as np
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import select_device

    if not args.model.is_file() or not args.input.is_file():
        raise FileNotFoundError(f'Check model/video paths: {args.model}, {args.input}')
    requested_device = args.device
    args.device = resolve_device(args.device)
    try:
        select_device(args.device, verbose=False)
    except (ValueError, RuntimeError, AssertionError) as error:
        if args.device.lower() == 'cpu':
            raise
        print(f'Device {args.device} unavailable: {error}; using CPU.', flush=True)
        args.device = 'cpu'
    model = YOLO(str(args.model))
    if model.task != 'segment':
        raise ValueError('Use a segmentation checkpoint: both methods must receive the same YOLO masks.')
    names = model.names
    classes = [i for i, name in names.items() if name.lower() in COLORS]
    if not classes:
        raise ValueError(f'No vehicle classes in checkpoint: {names}')
    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {args.input}')
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if width <= 0 or height <= 0:
        raise ValueError('Invalid video dimensions')
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0
    algorithms = ['quickhull', 'ograham'] if args.algorithm == 'both' else [args.algorithm]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {a: args.output_dir / f'output_{args.input.stem}_{a}.mp4' for a in algorithms}
    csv_path = args.output_dir / f'comparison_{args.input.stem}_{args.algorithm}.csv'
    json_path = csv_path.with_suffix('.json')
    for path in [*paths.values(), csv_path, json_path]:
        if path.resolve() in (args.input.resolve(), args.model.resolve()):
            raise ValueError(f'Output must not replace input: {path}')
        if path.exists() and not args.overwrite:
            raise FileExistsError(f'Output exists: {path}; change --output-dir or use --overwrite')
    writers, csv_file, results = {}, None, None
    stats = {a: {'hull_ms': 0.0, 'hull_calls': 0, 'postprocess_ms': 0.0} for a in algorithms}
    frame_count, missing_masks, objects, tracked_objects = 0, 0, 0, 0
    pair_ious = []
    model_stage_ms = {'preprocess': 0.0, 'inference': 0.0, 'postprocess': 0.0}
    tracker = IoUTracker(iou_threshold=args.track_iou, max_age=args.track_max_age)
    tracking_ms = 0.0
    # Warm inference and NMS before measuring the video run.
    # This avoids counting cold MPS operator initialization as hull/inference time.
    warmup_start = perf_counter()
    warmup_cap = cv2.VideoCapture(str(args.input))
    ok, warmup_frame = warmup_cap.read()
    warmup_cap.release()
    if not ok:
        raise RuntimeError('Cannot decode first frame')
    model.predict(source=warmup_frame, classes=classes, conf=args.conf,
                  imgsz=args.imgsz, max_det=args.max_det, device=args.device,
                  verbose=False)
    warmup_seconds = perf_counter() - warmup_start
    started = perf_counter()
    print(f'Model: {args.model.name}, scale={model.model.yaml.get("scale")}, device={args.device}; video {width}x{height}, {source_frames} frames', flush=True)
    try:
        for a, path in paths.items():
            writers[a] = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
            if not writers[a].isOpened():
                raise RuntimeError(f'Cannot create video: {path}')
        csv_file = csv_path.open('w', newline='', encoding='utf-8')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['frame', 'object', 'track_id', 'class_id', 'class_name', 'confidence',
                             'algorithm', 'input_points', 'hull_ms', 'area', 'perimeter',
                             'bbox_fill_ratio', 'vertices', 'pair_mask_iou'])
        results = model.predict(source=str(args.input), classes=classes, conf=args.conf,
                              imgsz=args.imgsz, max_det=args.max_det, device=args.device,
                              stream=True,
                              verbose=False)
        for result in results:
            frame_count += 1
            for stage in model_stage_ms:
                model_stage_ms[stage] += result.speed.get(stage, 0.0) or 0.0
            frames = {a: result.orig_img.copy() for a in algorithms}
            counts = {i: 0 for i in classes}
            # One GPU-to-CPU transfer for all boxes, instead of per-object transfers.
            box_data = result.boxes.data.cpu().numpy() if result.boxes is not None else np.empty((0, 6))
            # Associate before drawing; each ID stays attached to its original
            # detection index, so segmentation masks cannot be assigned by track order.
            coordinates = box_data[:, :4].astype(np.int32)
            detections = [(bbox.tolist(), int(row[-1]), float(row[-2]))
                          for bbox, row in zip(coordinates, box_data)]
            t_track = perf_counter()
            tracks = tracker.update(detections)
            detection_ids = {t['detection_index']: t['id'] for t in tracks
                             if t['detection_index'] is not None}
            tracking_ms += (perf_counter() - t_track) * 1000
            tracked_objects += len(detection_ids)
            objects += len(detections)
            polygons = {}
            if result.masks is not None:
                mask_indices = sorted(i for i in detection_ids if i < len(result.masks))
                if mask_indices:
                    polygons = dict(zip(mask_indices, result.masks[mask_indices].xy))
            for index, (bbox, cls, score) in enumerate(detections):
                if index not in detection_ids:
                    continue
                counts[cls] += 1
                name = names[cls]
                color = COLORS[name.lower()]
                track_id = detection_ids.get(index, '')
                points = np.asarray(polygons[index], dtype=np.int32) if index in polygons else None
                hulls, rows = {}, {}
                if points is None or len(points) < 3:
                    missing_masks += 1
                else:
                    # Alternate method order to reduce first-method timing bias.
                    order = algorithms if (frame_count + index) % 2 else algorithms[::-1]
                    for a in order:
                        t0 = perf_counter()
                        hulls[a] = compute_hull(points, a)
                        elapsed = (perf_counter() - t0) * 1000
                        stats[a]['hull_ms'] += elapsed
                        stats[a]['hull_calls'] += 1
                        features = extract_vector_features(hulls[a], bbox)
                        stats[a]['postprocess_ms'] += (perf_counter() - t0) * 1000
                        rows[a] = [frame_count, index, track_id, cls, name, score, a,
                                   len(points), elapsed, *features]
                    pair_iou = ''
                    if len(hulls) == 2:
                        # Raster comparison excluded from both method timers.
                        x1, y1 = points.min(axis=0)
                        x2, y2 = points.max(axis=0)
                        masks = []
                        for a in algorithms:
                            mask = np.zeros((y2-y1+1, x2-x1+1), dtype=np.uint8)
                            cv2.fillPoly(mask, [(hulls[a] - [x1, y1]).reshape(-1, 1, 2)], 1)
                            masks.append(mask.astype(bool))
                        union = np.logical_or(*masks).sum()
                        pair_iou = float(np.logical_and(*masks).sum() / union) if union else 1.0
                        pair_ious.append(pair_iou)
                    for a in algorithms:
                        csv_writer.writerow(rows[a] + [pair_iou])
                label = f'{name} {score:.2f}'
                if track_id != '':
                    label = f'#{track_id} {label}'
                for a, frame in frames.items():
                    if a in hulls:
                        cv2.polylines(frame, [hulls[a].reshape(-1, 1, 2)], True, color, 2)
                        top_y = int(hulls[a][:, 1].min())
                    else:
                        cv2.rectangle(frame, tuple(bbox[:2]), tuple(bbox[2:]), color, 2)
                        top_y = bbox[1]
                    cv2.putText(frame, label, (max(0, bbox[0]), max(15, top_y-5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            for a, frame in frames.items():
                summary = '  '.join(f'{names[i]}:{counts[i]}' for i in classes)
                cv2.putText(frame, f'{a}  {summary}', (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2, cv2.LINE_AA)
                writers[a].write(frame)
                if args.show:
                    cv2.imshow(a, frame)
            if frame_count == 1 or frame_count % 30 == 0:
                print(f'frame {frame_count}/{source_frames}; {objects} detections; elapsed {perf_counter()-started:.1f}s', flush=True)
            if args.show and cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if args.max_frames and frame_count >= args.max_frames:
                break
    finally:
        if results is not None:
            results.close()
        for writer in writers.values():
            writer.release()
        if csv_file:
            csv_file.close()
        if args.show:
            cv2.destroyAllWindows()
    if frame_count == 0:
        raise RuntimeError('No frames processed')
    for s in stats.values():
        s['mean_hull_ms'] = s['hull_ms'] / max(1, s['hull_calls'])
    metadata = {'model': str(args.model.resolve()),
                'model_sha256': hashlib.sha256(args.model.read_bytes()).hexdigest(),
                'task': model.task, 'scale': model.model.yaml.get('scale'), 'names': names,
                'source': str(args.input.resolve()), 'frames': frame_count, 'source_frames': source_frames,
                'missing_masks': missing_masks, 'detections': objects,
                'tracked_detections': tracked_objects, 'algorithms': stats,
                'pair_mask_iou_mean': float(np.mean(pair_ious)) if pair_ious else None,
                'pair_mask_iou_min': min(pair_ious) if pair_ious else None,
                'model_stage_ms': model_stage_ms, 'wall_seconds': perf_counter()-started,
                'tracker': {'type': 'jetson_iou_motion', 'iou_threshold': args.track_iou,
                            'max_age': args.track_max_age, 'high_conf': tracker.high_conf,
                            'low_conf': tracker.low_conf, 'assignment': 'hungarian_per_class',
                            'tracks_created': tracker.next_id - 1,
                            'tracking_ms': tracking_ms,
                            'mean_tracking_ms_per_frame': tracking_ms / frame_count},
                'warmup_seconds': warmup_seconds,
                'settings': {'conf': args.conf, 'imgsz': args.imgsz, 'device': args.device,
                             'requested_device': requested_device, 'max_det': args.max_det},
                'note': 'Hull timers include shared normalization, exclude drawing/CSV/raster IoU/shared YOLO/IoU tracking. Pair IoU is algorithm agreement, not ground-truth accuracy.'}
    json_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: metadata[k] for k in ('frames', 'algorithms', 'pair_mask_iou_mean', 'wall_seconds')}, indent=2), flush=True)
    print(f'Outputs: {args.output_dir.resolve()}', flush=True)


def main(default_algorithm='quickhull', default_model=None, default_output=None):
    run(parse_args(default_algorithm, default_model, default_output))

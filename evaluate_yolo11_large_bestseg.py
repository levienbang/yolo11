"""Evaluate 100 Vietnam images with existing YOLO11 large bestseg and both hulls."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts'))
from evaluate_vietnam_segmentation import main

if __name__ == '__main__':
    main(ROOT / 'weights/yolo-large/bestseg.pt', ROOT / 'outputs/evaluation/vietnam-seg-review-100-large')

"""Evaluate the 100 Vietnam images with YOLO11n-seg (COCO) and every region output."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts'))
from evaluate_vietnam_segmentation import main

if __name__ == '__main__':
    main(ROOT / 'weights/yolo-nano/yolo11n-seg.pt', ROOT / 'outputs/evaluation/vietnam-seg-100-nano')

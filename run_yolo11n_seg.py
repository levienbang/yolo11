"""Video: YOLO11n-seg COCO, four vehicle classes, QuickHull and O-Graham."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from vehicle_pipeline.inference_common import main

if __name__ == '__main__':
    main('both', ROOT / 'weights/yolo-nano/yolo11n-seg.pt', ROOT / 'outputs/vh1-nano-coco')

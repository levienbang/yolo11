"""Video demo: YOLO11l-seg (COCO), four vehicle classes, O-QuickHull and O-Graham."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from vehicle_pipeline.inference_common import main

if __name__ == '__main__':
    main('both', ROOT / 'weights/yolo-large/yolo11l-seg.pt', ROOT / 'outputs/vh1-large')

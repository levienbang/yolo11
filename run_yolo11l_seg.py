"""Video demo: YOLO11l-seg (COCO), ByteTrack, QuickHull and Graham hulls."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from vehicle_pipeline.inference_common import main

if __name__ == '__main__':
    main('both', ROOT / 'weights/yolo-large/yolo11l-seg.pt', ROOT / 'outputs/traffic-night-large')

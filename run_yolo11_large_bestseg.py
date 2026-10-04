"""Video: existing YOLO11 large bestseg, four vehicle classes, both hulls."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from vehicle_pipeline.inference_common import main

if __name__ == '__main__':
    main('both', ROOT / 'weights/yolo-large/bestseg.pt', ROOT / 'outputs/vh1-large-bestseg')

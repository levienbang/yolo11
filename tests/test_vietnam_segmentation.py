from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from prepare_vietnam_segmentation import load_polygons
from evaluate_vietnam_segmentation import load_gt, region_iou, matching, summarize
from vehicle_pipeline.vehicle_benchmark import confusion


class VietnamSegmentationTests(unittest.TestCase):
    def test_polygon_gt_and_box_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'gt.txt'
            path.write_text('0 .1 .1 .6 .1 .6 .6 .1 .6\n')
            gt,shape=load_gt(path,(100,100))
            self.assertEqual(shape,(100,100))
            self.assertEqual(gt[0]['class'],0)
            self.assertAlmostEqual(region_iou(gt[0]['region'],gt[0]['region']),1)
            path.write_text('0 .5 .5 .2 .2\n')
            with self.assertRaisesRegex(ValueError,'Non-polygon'): load_gt(path,(100,100))
            path.write_text('0 .1 .1 .2 .2 .3 .3\n')
            with self.assertRaisesRegex(ValueError,'Zero-area'):load_gt(path,(100,100))

    def test_confusion_uses_wrong_class_overlaps_ap_does_not(self):
        record=dict(scores=[.9],pred_classes=[1],gt_classes=[0],
                    ious={'box_detection':np.array([[1.]]),'yolo_mask':np.array([[1.]])})
        cm,pairs=confusion(record,.2,.5)
        self.assertEqual(cm[0,1],1)
        self.assertEqual(pairs,{0:0})
        self.assertEqual(matching(record,'yolo_mask',.2)[1],{})
        metrics=summarize([record],'yolo_mask',.2)
        self.assertEqual((metrics['tp'],metrics['fp'],metrics['fn']),(0,1,1))
        self.assertEqual(metrics['map50'],0)

    def test_absent_bicycle_not_relabelled(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'gt.txt';path.write_text('0 .5 .5 .2 .2\n1 .1 .1 .6 .1 .6 .6 .1 .6\n')
            targets=load_polygons(path,{0:'bicycle',1:'truck'},(100,100))
            self.assertEqual(len(targets),1)
            self.assertEqual(targets[0]['class_id'],1)

if __name__=='__main__':unittest.main()

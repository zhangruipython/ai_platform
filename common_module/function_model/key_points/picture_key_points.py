# -*- coding: utf-8 -*-
"""
@Time    : 2020-03-17 15:35
@Author  : zhangrui
@FileName: picture_key_points.py
@Software: PyCharm
人体关键点检测
"""
from detectron2.utils.logger import setup_logger

setup_logger()
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from common_module.function_model.config import COMMON_CONFIGS

class KeyPoints:
    def __init__(self, input_mat_img, cfg_path=COMMON_CONFIGS["KeyPoints"]["CFG_FILE"],
                 model_path=COMMON_CONFIGS["KeyPoints"]["MODEL_FILE"]):
        self.input_mat_img = input_mat_img
        self.model_path = model_path
        self.cfg_path = cfg_path

    @staticmethod
    def create_text_labels(classes, scores, class_names):
        """
            Args:
                classes (list[int] or None):
                scores (list[float] or None):
                class_names (list[str] or None):

            Returns:
                list[str] or None
        """
        labels = None
        if classes is not None and class_names is not None and len(class_names) > 1:
            labels = [class_names[i] for i in classes]
        if scores is not None:
            if labels is None:
                labels = ["{:.0f}".format(s * 100) for s in scores]
            else:
                labels = [{l: "{:.0f}".format(s * 100)} for l, s in zip(labels, scores)]
        return labels

    def make_keypoints_check(self):
        """
        进行关键点检测
        :return:
        """
        model_cfg = get_cfg()
        model_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        model_cfg.merge_from_file(self.cfg_path)
        model_cfg.MODEL.WEIGHTS = self.model_path
        predictor = DefaultPredictor(model_cfg)
        predictions = predictor(self.input_mat_img)["instances"].to("cpu")
        v = Visualizer(self.input_mat_img, MetadataCatalog.get(model_cfg.DATASETS.TRAIN[0]), scale=1.2)
        vis = v.draw_instance_predictions(predictions)
        return vis.get_image()


if __name__ == '__main__':
    weight_cfg_path = "/home/detectron2/configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
    weight_path = "/root/.torch/fvcore_cache/detectron2/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x/137849621/model_final_a6e10b.pkl"
    image_path = "/home/image/person01.jpg"
    key_points = KeyPoints(model_path=weight_path, cfg_path=weight_cfg_path, img_path=image_path)
    key_points.make_keypoints_check()

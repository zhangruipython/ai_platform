# -*- coding: utf-8 -*-
"""
@Time    : 2020-03-13 14:00
@Author  : zhangrui
@FileName: picture_instance_segm.py
@Software: PyCharm
图片实例分割
"""

from detectron2.utils.logger import setup_logger

setup_logger()
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from object_detect_service.config import COMMON_CONFIGS


class InstanceSegmentation:
    def __init__(self, input_mat_img, model_path=COMMON_CONFIGS["InstanceSegmentation"]["MODEL_FILE"],
                 cfg_path=COMMON_CONFIGS["InstanceSegmentation"]["CFG_FILE"]):
        """
        :param model_path: 模型地址
        :param cfg_path: cfg配置文件地址
        :param input_mat_img: 传入mat格式图片
        """
        self.model_path = model_path
        self.cfg_path = cfg_path
        self.input_mat_img = input_mat_img

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
        # if scores is not None:
        #     if labels is None:
        #         labels = ["{:.0f}".format(s * 100) for s in scores]
        #     else:
        #         labels = [{l: "{:.0f}".format(s * 100)} for l, s in zip(labels, scores)]
        return labels

    @staticmethod
    def draw_picture(metadata, img, predictions):
        v = Visualizer(img, metadata)
        v = v.draw_instance_predictions(predictions)
        return v.get_image()

    def make_instance_segment(self):
        """
        返回识别参数
        :return:
        """
        # im = read_image(self.img_path)
        # im = cv2.imread(self.img_path)
        # cfg配置文件
        model_cfg = get_cfg()
        model_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        model_cfg.merge_from_file(self.cfg_path)
        model_cfg.MODEL.WEIGHTS = self.model_path
        predictor = DefaultPredictor(model_cfg)
        predictions = predictor(self.input_mat_img)["instances"].to("cpu")
        detect_scores = predictions.scores if predictions.has("scores") else None
        detect_classes = predictions.pred_classes if predictions.has("pred_classes") else None
        metadata = MetadataCatalog.get(model_cfg.DATASETS.TRAIN[0])
        # 检测结果写入图片
        output_mat_img = self.draw_picture(metadata, self.input_mat_img, predictions)
        label = self.create_text_labels(classes=detect_classes, scores=detect_scores,
                                        class_names=metadata.get("thing_classes", None))
        # key_points = predictions.pred_keypoints if predictions.has("pred_keypoints") else None
        return {"label": label, "output_mat_img": output_mat_img}


if __name__ == '__main__':
    weight_cfg_path = "/home/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    weight_path = "/root/.torch/fvcore_cache/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
    image_path = "/home/image/dog.jpg"
    output_path = "instance.jpg"
    instance_segmentation = InstanceSegmentation(model_path=weight_path, cfg_path=weight_cfg_path,)
    print(instance_segmentation.make_instance_segment())

# -*- coding: utf-8 -*-
"""
@Time    : 2020-03-17 11:29
@Author  : zhangrui
@FileName: picture_panoramic_segmentation.py
@Software: PyCharm
图片全景分割
"""
from detectron2.utils.logger import setup_logger

setup_logger()
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from common_module.function_model.config import COMMON_CONFIGS
import torch
import numpy as np


class _PanopticPrediction:
    def __init__(self, panoptic_seg, segments_info):
        self._seg = panoptic_seg

        self._sinfo = {s["id"]: s for s in segments_info}  # seg id -> seg info
        segment_ids, areas = torch.unique(panoptic_seg, sorted=True, return_counts=True)
        areas = areas.numpy()
        sorted_idxs = np.argsort(-areas)
        self._seg_ids, self._seg_areas = segment_ids[sorted_idxs], areas[sorted_idxs]
        self._seg_ids = self._seg_ids.tolist()
        for sid, area in zip(self._seg_ids, self._seg_areas):
            if sid in self._sinfo:
                self._sinfo[sid]["area"] = float(area)

    def non_empty_mask(self):
        """
        Returns:
            (H, W) array, a mask for all pixels that have a prediction
        """
        empty_ids = []
        for seg_id in self._seg_ids:
            if seg_id not in self._sinfo:
                empty_ids.append(seg_id)
        if len(empty_ids) == 0:
            return np.zeros(self._seg.shape, dtype=np.uint8)
        assert (
                len(empty_ids) == 1
        ), ">1 ids corresponds to no labels. This is currently not supported"
        return (self._seg != empty_ids[0]).numpy().astype(np.bool)

    def semantic_masks(self):
        for sid in self._seg_ids:
            sinfo = self._sinfo.get(sid)
            if sinfo is None or sinfo["isthing"]:
                # Some pixels (e.g. id 0 in PanopticFPN) have no instance or semantic predictions.
                continue
            yield (self._seg == sid).numpy().astype(np.bool), sinfo

    def instance_masks(self):
        for sid in self._seg_ids:
            sinfo = self._sinfo.get(sid)
            if sinfo is None or not sinfo["isthing"]:
                continue
            mask = (self._seg == sid).numpy().astype(np.bool)
            if mask.sum() > 0:
                yield mask, sinfo


class PanoramicSegmentation:
    def __init__(self, input_mat_img, model_path=COMMON_CONFIGS["PanoramicSegmentation"]["MODEL_FILE"],
                 cfg_path=COMMON_CONFIGS["PanoramicSegmentation"]["CFG_FILE"]):
        """

        :param input_mat_img: 传入mat格式图片
        :param model_path: 模型地址
        :param cfg_path: cfg配置文件地址
        """
        self.input_mat_img = input_mat_img
        self.model_path = model_path
        self.cfg_path = cfg_path
        self.panoptic_seg = []
        self.segments_info = []
        self.cfg = []

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

    def make_panoramic_segment(self):
        """
        全景分割图片
        :return: {"labels":['bicycle', 'dog', 'car'],"output_mat_img":mat格式图片}
        """
        # im = read_image(self.image_path)
        model_cfg = get_cfg()
        model_cfg.merge_from_file(self.cfg_path)
        model_cfg.MODEL.WEIGHTS = self.model_path
        predictor = DefaultPredictor(model_cfg)
        panoptic_seg, segments_info = predictor(self.input_mat_img)["panoptic_seg"]
        pred = _PanopticPrediction(panoptic_seg.to("cpu"), segments_info)
        all_instances = list(pred.instance_masks())
        masks, sinfo = list(zip(*all_instances))
        category_ids = [x["category_id"] for x in sinfo]
        try:
            scores = [x["scores"] for x in sinfo]
        except KeyError:
            scores = None
        metadata = MetadataCatalog.get(model_cfg.DATASETS.TRAIN[0])
        labels = self.create_text_labels(category_ids, scores, metadata.thing_classes)
        v = Visualizer(self.input_mat_img, MetadataCatalog.get(model_cfg.DATASETS.TRAIN[0]),
                       scale=1.2)
        vis = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
        return {"labels": labels, "output_mat_img": vis.get_image()[:, :, ::-1]}


if __name__ == '__main__':
    img_path = "/home/image/dog.jpg"
    weight_cfg_path = "/home/detectron2/configs/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"
    weight_path = "/root/.torch/fvcore_cache/detectron2/COCO-PanopticSegmentation/model_final_cafdb1.pkl"
    # panoramic_segmentation = PanoramicSegmentation(image_path=img_path, model_path=weight_path,
    #                                                cfg_path=weight_cfg_path)
    # panoramic_segmentation.make_panoramic_segment()
    # print(panoramic_segmentation.get_param())
    # print(panoramic_segmentation.draw_image())

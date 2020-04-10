# -*- coding: utf-8 -*-
"""
@Time    : 2019-11-26 13:31
@Author  : zhangrui
@FileName: detect_picture_app.py
@Software: PyCharm
"""
from ai_rpc_service.common_util_module import do_detect


class DetectImage:
    def __init__(self, model_param, zh_en_dir, confidence_threshold):
        self.model_param = model_param
        self.zh_en_dir = zh_en_dir
        self.confidence_threshold = confidence_threshold

    def detect(self, image_path):
        detect_param = do_detect.detection(self.model_param[3], self.model_param[0], self.model_param[1],
                                           self.model_param[2], image_path)
        if detect_param["detections"]:
            detect_param_box = do_detect.compute_detect_box(detections=detect_param["detections"],
                                                            img=detect_param["img"], zh_en_dir=self.zh_en_dir,
                                                            confidence_threshold=self.confidence_threshold)
            return detect_param_box
        else:
            return {"detection_coo_list": [], "img": detect_param["img"]}

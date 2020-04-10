# -*- coding: utf-8 -*-
"""
@Time    : 2020-03-31 15:27
@Author  : zhangrui
@FileName: picture_mask_detect.py
@Software: PyCharm
"""
from common_module.base_tool import do_detect, model_load
from common_module.function_model.config import COMMON_CONFIGS
import cv2
import uuid
import os

CONFIDENCE = 0.4


class MaskDetect:
    def __init__(self, input_mat_img, model_path=COMMON_CONFIGS["MaskDetect"]["MODEL_FILE"],
                 cfg_path=COMMON_CONFIGS["MaskDetect"]["CFG_FILE"],
                 names_path=COMMON_CONFIGS["MaskDetect"]["NAMES_FILE"],
                 darknet_path=COMMON_CONFIGS["MaskDetect"]["DARKNET_FILE"]):
        self.input_mat_img = input_mat_img
        self.model_path = model_path
        self.cfg_path = cfg_path
        self.names_path = names_path
        self.darknet_path = darknet_path

    def make_detect(self):
        img_name = str(uuid.uuid4()) + ".jpg"
        cv2.imwrite(img_name, self.input_mat_img)
        model_param = model_load.load_model(darknet_path=self.darknet_path, configPath=self.cfg_path,
                                            weightPath=self.model_path, metaPath=self.names_path)
        detect_param_box = do_detect.detection(model_param[3], model_param[0], model_param[1], model_param[2], img_name)
        param = do_detect.compute_detection(detections=detect_param_box["detections"], img=detect_param_box["img"],
                                            confidence_threshold=CONFIDENCE)
        os.remove(img_name)
        return do_detect.draw_detect_img(param)


if __name__ == '__main__':
    mat_img = cv2.imread("/home/detectron2/mask01.jpeg")
    mask_detect = MaskDetect(input_mat_img=mat_img)
    mat_img_out = mask_detect.make_detect()
    cv2.imwrite("/home/detectron2/mask_detect.jpg", mat_img_out)

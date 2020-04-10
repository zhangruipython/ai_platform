# -*- coding: utf-8 -*-
"""
@Time    : 2020-01-15 15:32
@Author  : zhangrui
@FileName: stitch_count_control.py
@Software: PyCharm
"""
import sys

sys.path.append('../../../')
sys.path.append('../../')
from common_module.base_tool import model_load
from ai_rpc_service.object_count_service.tools import zh_en_dir
from common_module.picture_stitch_module import img_stitching, detect_picture
import multiprocessing
import cv2
import uuid


class DetectStitchCount:
    def __init__(self, video_path, confidence, yolo_model_name):
        self.video_path = video_path
        self.confidence = confidence
        self.yolo_model_name = yolo_model_name
        self.zh_en = zh_en_dir.zh_en_map(yolo_model_name)

    def detect_stitch(self, return_dict, split_len=40):
        """
        拼接图片识别
        :param return_dict:
        :param split_len: 取出视频拼接图片间隔
        :return: 识别参数
        """
        yolo_model_param = model_load.load(config_path="../../common_util_module/config.conf",
                                           model_name=self.yolo_model_name)
        picture_stitch = img_stitching.MorePictureJoint(video_path=self.video_path, split_len=split_len)
        stitching_img = picture_stitch.stitch_img()
        predict_path = str(uuid.uuid4()) + ".jpg"
        cv2.imwrite(predict_path, stitching_img)
        detect_image = detect_picture.DetectImage(model_param=yolo_model_param, zh_en_dir=zh_en_dir,
                                                  confidence_threshold=self.confidence)
        detect_param_list = detect_image.detect(predict_path)["detection_coo_list"]
        print("detect_param_list{a}".format(a=detect_param_list))
        return_dict["detect_param_list"] = detect_param_list

    def control_process(self):
        manage = multiprocessing.Manager()
        return_dict = manage.dict()
        track_process = multiprocessing.Process(target=self.detect_stitch, args=(return_dict, 40))
        track_process.start()
        track_process.join()
        return return_dict["detect_param_list"]

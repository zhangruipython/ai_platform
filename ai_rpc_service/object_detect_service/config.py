# -*- coding: utf-8 -*-
"""
@Time    : 2020-03-30 16:03
@Author  : zhangrui
@FileName: config.py
@Software: PyCharm
配置文件
"""
# 常用类别检测模型配置
COMMON_CONFIGS = {
    # 实例分割配置文件
    "InstanceSegmentation": {
        "MODEL_FILE": "/root/.torch/fvcore_cache/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl",
        "CFG_FILE": "/home/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    },
    # 人体关键点检测配置文件
    "KeyPoints": {
        "MODEL_FILE": "/root/.torch/fvcore_cache/detectron2/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x/137849621/model_final_a6e10b.pkl",
        "CFG_FILE": "/home/detectron2/configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
    },
    # 全景分割
    "PanoramicSegmentation": {
        "MODEL_FILE": "/root/.torch/fvcore_cache/detectron2/COCO-PanopticSegmentation/model_final_cafdb1.pkl",
        "CFG_FILE": "/home/detectron2/configs/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"
    },
    # 口罩佩戴检测
    "MaskDetect": {
        "MODEL_FILE": "/home/detectron2/backup/mask_21000.weights",
        "CFG_FILE": "/home/detectron2/configs/mask_detect_yolo/mask_yolov3.cfg",
        "NAMES_FILE": "/home/detectron2/configs/mask_detect_yolo/mask.data",
        "DARKNET_FILE": "/home/darknet-master"
    }
}

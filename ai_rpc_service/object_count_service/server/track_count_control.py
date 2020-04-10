# -*- coding: utf-8 -*-
"""
@Time    : 2020-01-15 14:26
@Author  : zhangrui
@FileName: track_count_control.py
@Software: PyCharm
将跟踪盘点服务添加至进程中通过进程管控，实现手动释放显存
（如果直接在服务端调用跟踪盘点方法会出现，即使对象生命周期结束依然不释放显存的现象）
"""
import sys

sys.path.append('../../../')
sys.path.append('../')
from common_module.base_tool import do_detect, model_load
from ai_rpc_service.object_count_service.tools import zh_en_dir
from common_module.deep_sort_module import nn_matching
from common_module.deep_sort_module import preprocessing
from common_module.deep_sort_module.tracker import Tracker
from common_module.deep_sort_module.detection import Detection
from ai_rpc_service.object_count_service.tools import generate_detections
import numpy as np
import multiprocessing
import cv2
import uuid


class DetectTrackCount:
    # 初始化deep_sort参数
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0
    model_filename = '/home/hadoop/RongzerAI/rongerai/property_check/model_data/mars-small128.pb'

    def __init__(self, video_path, confidence, yolo_model_name):
        self.video_path = video_path
        self.confidence = confidence
        self.yolo_model_name = yolo_model_name
        self.frame_path = str(uuid.uuid4()) + ".jpg"
        self.zh_en = zh_en_dir.zh_en_map(yolo_model_name)

    def detect_track(self, return_dict):
        property_count = []
        encoder = generate_detections.create_box_encoder(self.model_filename, batch_size=1)
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", self.max_cosine_distance, self.nn_budget)
        yolo_model_param = model_load.load(config_path="../../common_util_module/config.conf",
                                           model_name=self.yolo_model_name)
        tracker = Tracker(metric)
        video_capture = cv2.VideoCapture(self.video_path)
        while True:
            ret, frame = video_capture.read()
            if not ret:
                video_capture.release()
                break
            else:
                cv2.imwrite(self.frame_path, frame)
                detect_param = do_detect.detection(yolo_model_param[3], yolo_model_param[0],
                                                   yolo_model_param[1], yolo_model_param[2],
                                                   self.frame_path)
                label_boxes, label_names = do_detect.detection_to_tracker(detect_param["detections"], self.confidence,
                                                                          self.zh_en)
                features = encoder(frame, label_boxes)
                detections = [Detection(bbox, 1.0, feature, label_name) for bbox, feature, label_name in
                              zip(label_boxes, features, label_names)]
                boxes = np.array([d.tlwh for d in detections])
                scores = np.array([d.confidence for d in detections])
                # 识别区非极大值抑制
                indices = preprocessing.non_max_suppression(boxes, self.nms_max_overlap, scores)
                all_detections = [detections[i] for i in indices]
                tracker.predict()
                # 跟踪器更新，返回新识别类别名称
                update_label_names = tracker.update(all_detections)
                property_count += update_label_names
        video_capture.release()
        return_dict["property_count"] = property_count

    def control_process(self):
        manage = multiprocessing.Manager()
        return_dict = manage.dict()
        track_process = multiprocessing.Process(target=self.detect_track, args=(return_dict,))
        track_process.start()
        track_process.join()
        return return_dict["property_count"]

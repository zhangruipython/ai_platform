# -*- coding: utf-8 -*-
"""
@Time    : 2020-01-09 14:39
@Author  : zhangrui
@FileName: stitch_count_server.py
@Software: PyCharm
"""
import sys

sys.path.append('../../../')
sys.path.append('../../')
from ai_rpc_service.object_count_service import object_count_pb2
from ai_rpc_service.object_count_service import object_count_pb2_grpc
from ai_rpc_service.object_count_service.server import stitch_count_control
from concurrent import futures
from collections import Counter
import grpc


class StitchCountService(object_count_pb2_grpc.ObjectCountServicer):
    def StitchObjectCount(self, request, context):
        video_path = request.video_path
        confidence_threshold = request.detect_confidence
        yolo_model_name = request.model_name
        detect_stitch_count = stitch_count_control.DetectStitchCount(video_path=video_path,
                                                                     confidence=confidence_threshold,
                                                                     yolo_model_name=yolo_model_name)
        detect_param_list = detect_stitch_count.control_process()
        if detect_param_list:
            label_count = Counter([a["detect_label"] for a in detect_param_list]).most_common()
            label_count_list = [{"label": b[0], "count": b[1]} for b in label_count]
        else:
            label_count_list = []
        return object_count_pb2.ObjectCountResponse(object_param=label_count_list)


def StitchCountServe():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))
    object_count_pb2_grpc.add_ObjectCountServicer_to_server(StitchCountService(), server)
    server.add_insecure_port('[::]:5003')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    StitchCountServe()

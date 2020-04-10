# -*- coding: utf-8 -*-
"""
@Time    : 2020-01-09 11:00
@Author  : zhangrui
@FileName: object_count_client.py
@Software: PyCharm
"""
import sys

sys.path.append("../../../")
sys.path.append('../../')
from object_count_service import object_count_pb2
from object_count_service import object_count_pb2_grpc
import grpc


def TrackCountRun():
    with grpc.insecure_channel('localhost:5004') as channel:
        route_sub = object_count_pb2_grpc.ObjectCountStub(channel)
        response = route_sub.TrackObjectCount(object_count_pb2.ObjectCountRequest(video_path="/home/hadoop/Documents"
                                                                                             "/darknet-master-1"
                                                                                             "/darknet-master"
                                                                                             "/test_file/cut2.mp4",
                                                                                  detect_confidence=0.55,
                                                                                  model_name="yike_handle_model"))
    normalize_params = [{"object_label": param.object_label, "object_num": param.object_num}
                        for param in response.object_param]
    print(normalize_params)


def StitchCountRun():
    with grpc.insecure_channel('localhost:5003') as channel:
        route_sub = object_count_pb2_grpc.ObjectCountStub(channel)
        response = route_sub.StitchObjectCount(object_count_pb2.ObjectCountRequest(video_path="/home/hadoop/Documents"
                                                                                              "/darknet-master-1"
                                                                                              "/darknet-master"
                                                                                              "/test_file/demo07.mp4",
                                                                                   detect_confidence=0.55,
                                                                                   model_name="yike_handle_model"))
    normalize_params = [{"object_label": param.object_label, "object_num": param.object_num}
                        for param in response.object_param]
    print(normalize_params)


if __name__ == '__main__':
    TrackCountRun()

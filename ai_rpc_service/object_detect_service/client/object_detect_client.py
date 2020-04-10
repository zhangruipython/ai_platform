# -*- coding: utf-8 -*-
"""
@Time    : 2020-01-06 14:57
@Author  : zhangrui
@FileName: object_detect_client.py
@Software: PyCharm
"""
import grpc
from ai_rpc_service.object_detect_service import object_detect_pb2
from ai_rpc_service.object_detect_service import object_detect_pb2_grpc


def DetectPictureRun():
    with grpc.insecure_channel('localhost:5001') as channel:
        route_stub = object_detect_pb2_grpc.RouteGuideStub(channel)
        response = route_stub.DetectPicture(
            object_detect_pb2.RequestDetectPicture(picture_path="/home/hadoop/Documents/darknet-master-1/darknet-master"
                                                                "/test_file/computer01.jpg", detect_confidence=0.5,
                                                   model_name="coco_model"))

    normalize_params = [{"object_name": param.object_name, "object_confidence": param.object_confidence,
                         "lt_box": (param.lt_box.lt_x, param.lt_box.lt_y),
                         "rl_box": (param.rl_box.rl_x, param.rl_box.rl_y)} for param in response.detect_param]
    return normalize_params


def DemoRun():
    with grpc.insecure_channel("localhost:5002") as channel:
        stub = object_detect_pb2_grpc.RouteGuideStub(channel)
        request = object_detect_pb2.HelloRequest(greeting="zhangrui")
        response = stub.SayHello(request)
    print(response)


if __name__ == '__main__':
    DetectPictureRun()

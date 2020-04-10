# -*- coding: utf-8 -*-
"""
@Time    : 2020-01-06 14:19
@Author  : zhangrui
@FileName: object_detect_client.py
@Software: PyCharm
"""
import grpc
import sys
sys.path.append("../../../")
from ai_rpc_service.object_detect_service import object_detect_pb2
from ai_rpc_service.object_detect_service import object_detect_pb2_grpc
from concurrent import futures
from common_module.base_tool import do_detect, model_load


class DetectPictureService(object_detect_pb2_grpc.RouteGuideServicer):
    def __init__(self, config_path, model_name):
        self.model_param = model_load.load(config_path=config_path, model_name=model_name)

    def DetectPicture(self, request, context):
        img_path = request.picture_path
        detect_confidence = request.detect_confidence
        detect_param_box = do_detect.detection(self.model_param[3], self.model_param[0], self.model_param[1],
                                               self.model_param[2], img_path)
        param = do_detect.normalize_detect_param(detections=detect_param_box["detections"],
                                                 confidence_limit=detect_confidence)
        return object_detect_pb2.ResponseDetectPicture(detect_param=param)


class HelloService(object_detect_pb2_grpc.RouteGuideServicer):
    def SayHello(self, request, context):
        return object_detect_pb2.HelloResponse(reply="how are you")


def DetectPictureServe():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))
    object_detect_pb2_grpc.add_RouteGuideServicer_to_server(
        DetectPictureService(config_path="../../common_util_module/config.conf", model_name="coco_model"),
        server)
    server.add_insecure_port('[::]:5001')
    server.start()
    server.wait_for_termination()


def DemoServe():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=3))
    object_detect_pb2_grpc.add_RouteGuideServicer_to_server(HelloService(), server)
    server.add_insecure_port('[::]:5002')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    DetectPictureServe()

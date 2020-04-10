# -*- coding: utf-8 -*-
"""
@Time    : 2020-01-09 11:00
@Author  : zhangrui
@FileName: object_count_client.py
@Software: PyCharm
将服务添加至consul消息注册中心
"""
import sys
sys.path.append('../../../')
sys.path.append('../')
from ai_rpc_service.object_count_service import object_count_pb2
from ai_rpc_service.object_count_service import object_count_pb2_grpc
from ai_rpc_service.object_count_service.server import track_count_control
from concurrent import futures
from collections import Counter
import grpc


class TrackCountService(object_count_pb2_grpc.ObjectCountServicer):
    def TrackObjectCount(self, request, context):
        video_path = request.video_path
        confidence_threshold = request.detect_confidence
        yolo_model_name = request.model_name
        detect_track_count = track_count_control.DetectTrackCount(video_path=video_path,
                                                                  confidence=confidence_threshold,
                                                                  yolo_model_name=yolo_model_name)
        property_count = detect_track_count.control_process()
        print("property_count{a}".format(a=property_count))
        label_count = Counter(property_count).most_common()
        if label_count:
            label_count_list = [{"object_label": b[0], "object_num": b[1]} for b in label_count]
        else:
            label_count_list = []
        return object_count_pb2.ObjectCountResponse(object_param=label_count_list)


# def register(server_name, server_port, server_ip, register_ip):
#     """
#     服务注册
#     :param server_name: 服务名称
#     :param server_port: 服务端口
#     :param server_ip: 服务地址
#     :param register_ip: 注册中心地址
#     :return: None
#     """
#     register_consul = consul.Consul(host=register_ip)
#     register_check = consul.Check.tcp(host=server_ip, port=server_port, interval="10s")
#     register_consul.agent.service.register(name=server_name, service_id=f"{server_name}-{server_ip}-{server_port}",
#                                            port=server_port, address=server_ip, check=register_check)
#     print(f"{server_name}-{server_ip}-{server_port}服务注册成功")


# def unregister(server_name, server_ip, server_port):
#     unregister_consul = consul.Consul()
#     unregister_consul.agent.service.deregister(f"{server_name}-{server_ip}-{server_port}")
#     print(f"{server_name}-{server_ip}-{server_port}服务取消注册成功")


def TrackCountServe():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))
    object_count_pb2_grpc.add_ObjectCountServicer_to_server(TrackCountService(), server)
    server.add_insecure_port('[::]:5004')
    # register(server_name="track_count_server", server_port=5004, server_ip="192.168.1.54", register_ip="192.168.1.54")
    server.start()
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        # unregister(server_name="track_count_server", server_ip="192.168.1.54", server_port=5004)
        server.stop()


if __name__ == '__main__':
    TrackCountServe()

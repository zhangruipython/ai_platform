# -*- coding: utf-8 -*-
"""
@Time    : 2020-04-10 11:15
@Author  : zhangrui
@FileName: video_instance_segm.py
@Software: PyCharm
视频实例分割
"""

from common_module.function_model.config import COMMON_CONFIGS
from common_module.function_model.instance_segmentation import picture_instance_segm
import cv2
import time
import multiprocessing


class VideoInstanceSeg:
    def __init__(self, input_video_path, output_video_path,
                 model_path=COMMON_CONFIGS["InstanceSegmentation"]["MODEL_FILE"],
                 cfg_path=COMMON_CONFIGS["InstanceSegmentation"]["CFG_FILE"]):
        """

        :param input_video_path: 传入视频路径
        :param output_video_path: 输出视频路径
        :param model_path: 模型地址
        :param cfg_path: 模型配置文件地址
        """
        self.input_video_path = input_video_path
        self.output_video_path = output_video_path
        self.model_path = model_path
        self.cfg_path = cfg_path

    def read_video(self, return_dict, record_file):
        """
        读取视频，实例分割，记录label
        :param record_file: 记录识别label地址
        :param return_dict:
        :return:
        """
        capture = cv2.VideoCapture(self.input_video_path)
        out = cv2.VideoWriter(self.output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), 10.0,
                              (int(capture.get(3)), int(capture.get(4))))
        instance_seg = picture_instance_segm.InstanceSegmentation()
        i = 1
        with open(record_file, "a") as f:
            while capture.isOpened():
                success, frame = capture.read()
                if success:
                    frame_mes = instance_seg.make_instance_segment(frame)
                    output_mat_img = frame_mes["output_mat_img"]
                    out.write(output_mat_img)
                    labels_str = ""
                    for mes in frame_mes["label"]:
                        labels_str += (mes + ",")
                    f.write(str(i) + "," + labels_str + "\n")
                    i += 1
                else:
                    capture.release()
                    break

        capture.release()
        out.release()
        return_dict["output_video_path"] = self.output_video_path
        # return self.output_video_path

    def control_process(self, record_path):
        manage = multiprocessing.Manager()
        return_dict = manage.dict()
        instance_process = multiprocessing.Process(target=self.read_video, args=(return_dict,record_path))
        instance_process.start()
        instance_process.join()
        return return_dict["output_video_path"]


if __name__ == '__main__':
    input_path = "/home/video_file/red5/20200403/camera01_1585881723115.mp4"
    out_path = "/home/video_file/instance_video_dir/a.mp4"
    video_instance_seg = VideoInstanceSeg(input_video_path=input_path, output_video_path=out_path)
    time01 = time.time()
    a = video_instance_seg.control_process(record_path="/home/record.txt")
    print("耗时{a}".format(a=time.time() - time01))

# -*- coding: utf-8 -*-
"""
@Time    : 2020-03-30 10:12
@Author  : zhangrui
@FileName: flaskr_web.py
@Software: PyCharm
"""
import sys

sys.path.append("../../")
from flask import Flask
from flask import request
from flask import jsonify
from common_module.base_tool.image_base64 import ImgConversion
from common_module.function_model.instance_segmentation.picture_instance_segm import InstanceSegmentation
from common_module.function_model.panoramic_segmentation.picture_panoramic_segmentation import \
    PanoramicSegmentation
from common_module.function_model.key_points.picture_key_points import KeyPoints
from common_module.function_model.mask_detect.picture_mask_detect import MaskDetect

app = Flask(__name__)
img_conversion = ImgConversion()


@app.route("/hello/", methods=["post"])
def hello():
    json_data = request.get_json()
    name = json_data["name"]
    return jsonify(hello="hello" + name)


@app.route("/object_detect/instance_segm/", methods=["post"])
def instance_segm():
    """
    实例分割
    :return:{"base64_str":base64_str,"label":["dogs",]}
    """
    json_data = request.get_json()
    img_base_str = json_data["img_base64"]
    img_mat = img_conversion.base64_cv2(img_base_str)
    instance_segmentation = InstanceSegmentation(input_mat_img=img_mat)
    instance_param = instance_segmentation.make_instance_segment()
    print("detect over")
    base64_str = img_conversion.cv2_base64(mat_img=instance_param["output_mat_img"])
    return jsonify(base64_str=base64_str, labels=instance_param["label"])


@app.route("/object_detect/key_points/", methods=["post"])
def key_points():
    """
    关键点检测(关键点坐标参数获取=>visualizer.py=>overlay_instances=>draw_and_connect_keypoints)
    :return:{"base64_str":base64_str}
    """
    json_data = request.get_json()
    img_base_str = json_data["img_base64"]
    img_mat = img_conversion.base64_cv2(img_base_str)
    keypoints = KeyPoints(input_mat_img=img_mat)
    keypoints_param = keypoints.make_keypoints_check()
    print("detect over")
    base64_str = img_conversion.cv2_base64(mat_img=keypoints_param)
    return jsonify(base64_str=base64_str)


@app.route("/object_detect/panoramic_segm/", methods=["post"])
def panoramic_segm():
    """
    全景分割
    :return:{"base64_str":base64_str,"label":["dogs",]}
    """
    json_data = request.get_json()
    img_base_str = json_data["img_base64"]
    img_mat = img_conversion.base64_cv2(img_base_str)
    panoramic_segmentation = PanoramicSegmentation(input_mat_img=img_mat)
    panoramic_param = panoramic_segmentation.make_panoramic_segment()
    print("detect over")
    base64_str = img_conversion.cv2_base64(mat_img=panoramic_param["output_mat_img"])
    return jsonify(base64_str=base64_str, labels=panoramic_param["labels"])


@app.route("/object_detect/mask_detect/", methods=["post"])
def mask_detect():
    """
    口罩检测
    :return:{"base64_str":base64_str}
    """
    json_data = request.get_json()
    img_base_str = json_data["img_base64"]
    img_mat = img_conversion.base64_cv2(img_base_str)
    mask_detection = MaskDetect(input_mat_img=img_mat)
    mask_param = mask_detection.make_detect()
    print("detect over")
    base64_str = img_conversion.cv2_base64(mat_img=mask_param)
    return jsonify(base64_str=base64_str)


@app.route("/object_detect/video_instance_segm", methods=["post"])
def video_instance_segm():
    """
    视频实例分割
    :return: {"output_video_path":output_video_path}
    """
    json_data = request.get_json()
    input_video_path = json_data["input_video_path"]
    # 宿主机地址与容器地址映射
    container_path = "/home/video_file/instance_video_dir/instance_"
    host_machine_path = "/home/media-server/media-file/instance_video_dir/instance_"
    container_input_video_path = input_video_path.replace(VOLUME_MAP["VideoFileMap"]["HOST_MACHINE_PATH"],
                                                          VOLUME_MAP["VideoFileMap"]["CONTAINER_PATH"])
    container_output_video_path = container_path + input_video_path.split("/")[-1]
    container_output_record_path = container_output_video_path.split(".")[0] + ".txt"
    host_machine_output_video_path = host_machine_path + input_video_path.split("/")[-1]
    host_machine_output_record_path = host_machine_output_video_path.split(".")[0] + ".txt"
    video_instance_seg = VideoInstanceSeg(input_video_path=container_input_video_path,
                                          output_video_path=container_output_video_path)
    a = video_instance_seg.control_process(record_path=container_output_record_path)
    print("detect over")
    return jsonify(output_video_path=host_machine_output_video_path,
                   output_label_path=host_machine_output_record_path)

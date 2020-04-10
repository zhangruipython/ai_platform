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

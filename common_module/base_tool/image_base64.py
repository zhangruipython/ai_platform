# -*- coding: utf-8 -*-
"""
@Time    : 2020-03-30 15:14
@Author  : zhangrui
@FileName: image_base64.py
@Software: PyCharm
base64和图像互转
"""
import base64
import cv2
import numpy as np


class ImgConversion:
    @staticmethod
    def cv2_base64(mat_img):
        """
        cv2格式=>base64格式=>str
        :param mat_img:
        :return:
        """
        base64_str = cv2.imencode(".jpg", mat_img)[1]
        base64_code = base64.b64encode(base64_str)
        return str(base64_code, encoding="utf-8")

    @staticmethod
    def base64_cv2(base_img):
        """
        字符串=>base64=>cv2
        :param base_img:
        :return:
        """
        b64_data = base_img.split(",")[1]
        img_string = base64.b64decode(b64_data)
        np_arr = np.fromstring(img_string, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return image

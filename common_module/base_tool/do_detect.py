# -*- coding: utf-8 -*-
"""
@Time    : 2019-11-26 13:26
@Author  : zhangrui
@FileName: do_detect.py
@Software: PyCharm
GPU darknet识别图像
"""
import os
import sys
import cv2
from PIL import Image, ImageDraw, ImageFont

netMain = None
metaMain = None
altNames = None


def performDetect(darknet_path, configPath, weightPath, metaPath, imagePath, thresh=0.25):
    sys.path.append(darknet_path)
    import darknet
    global metaMain, netMain, altNames
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" + os.path.abspath(configPath) + "`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" + os.path.abspath(weightPath) + "`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" + os.path.abspath(metaPath) + "`")

    # 默认batch为1
    netMain = darknet.load_net_custom(configPath.encode("ascii"), weightPath.encode("ascii"), 0, 1)
    metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents, re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception as e:
            print(e)
    # 判断图片路径是否存在
    if not os.path.exists(imagePath):
        raise ValueError("Invalid image path `" + os.path.abspath(imagePath) + "`")
    # 开始预测
    detections = darknet.detect(netMain, metaMain, imagePath.encode("ascii"), thresh)
    return detections


# 加载模型[netMain, metaMain, thresh]
def load_model(darknet_path, configPath, weightPath, metaPath, thresh=0.25):
    sys.path.append(darknet_path)
    import darknet
    global metaMain, netMain, altNames
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" + os.path.abspath(configPath) + "`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" + os.path.abspath(weightPath) + "`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" + os.path.abspath(metaPath) + "`")

    # 默认batch为1
    netMain = darknet.load_net_custom(configPath.encode("ascii"), weightPath.encode("ascii"), 0, 1)
    metaMain = darknet.load_meta(metaPath.encode("ascii"))
    return netMain, metaMain, thresh, darknet_path


# 具体识别
def detection(darknet_path, net_main, meta_main, thresh, image_path):
    """
    通过传入模型预加载参数进行图片目标识别
    :param darknet_path:darknet.py文件所在地址
    :param net_main:模型加载文件
    :param meta_main:模型加载文件
    :param thresh:阈值
    :param image_path:识别图像地址
    :return:{"detections": detections, "img": 图像矩阵}
    如果不存在识别区detections为[]
    """
    sys.path.append(darknet_path)
    import darknet
    frame_sized = cv2.imread(image_path)
    image_path = image_path.encode("ascii")
    detections = darknet.detect(net_main, meta_main, image_path, thresh=thresh)
    return {"detections": detections, "img": frame_sized}


def convertBack(x, y, w, h):
    x_min = (round(x - (w / 2.0)))
    x_max = (round(x + (w / 2.0)))
    y_min = (round(y - (h / 2.0)))
    y_max = (round(y + (h / 2.0)))
    return x_min, y_min, x_max, y_max


def draw_detection(detection_param):
    """
    矩形框框选识别区 (针对中文标签无法通过cv2写入图片，所以使用PIL)
    :param detection_param:识别参数
    :return:
    """
    if detection_param["detection_coo_list"]:
        for param in detection_param["detection_coo_list"]:
            cv2.rectangle(detection_param["img"], param["pt1"], param["pt2"], (0, 255, 0), 3)
        pil_img = Image.fromarray(cv2.cvtColor(detection_param["img"], cv2.COLOR_RGB2BGR))
        img_w, img_h = detection_param["img"].shape[:2]
        if img_w > 3000:
            for param in detection_param["detection_coo_list"]:
                draw = ImageDraw.Draw(pil_img)
                front_style = ImageFont.truetype(font="/home/hadoop/Documents/SIMYOU.TTF", size=120, encoding="utf-8")
                draw.text((param["pt1"][0], param["pt1"][1] - 5), param["detect_label"], (255, 0, 0), front_style)
        else:
            for param in detection_param["detection_coo_list"]:
                draw = ImageDraw.Draw(pil_img)
                front_style = ImageFont.truetype(font="/home/hadoop/Documents/SIMYOU.TTF", size=30, encoding="utf-8")
                draw.text((param["pt1"][0], param["pt1"][1] - 5), param["detect_label"], (255, 0, 0), front_style)
            # cv2_img = cv2.cvtColor(numpy.asarray(pil_img), cv2.COLOR_RGB2BGR)
            # cv2.imwrite(predicted_picture_name, cv2_img)
    else:
        pil_img = Image.fromarray(cv2.cvtColor(detection_param["img"], cv2.COLOR_RGB2BGR))
    return pil_img


def draw_detect_img(detection_param):
    """
    矩形框框选识别区 (使用cv2绘制)
    :param detection_param: 识别参数
    :return: mat格式图片
    """
    if detection_param["detection_coo_list"]:
        for param in detection_param["detection_coo_list"]:
            cv2.rectangle(detection_param["img"], param["pt1"], param["pt2"], (0, 255, 0), 1)
            cv2.putText(detection_param["img"], param["detect_label"],
                        (param["pt1"][0], param["pt1"][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        [0, 255, 0], 1)
    return detection_param["img"]


# 处理识别参数
def compute_detect_box(detections, img, zh_en_dir, confidence_threshold):
    detection_coo_list = []
    for detection_param in detections:
        if detection_param[1] > confidence_threshold:
            x, y, w, h = detection_param[2][0], detection_param[2][1], detection_param[2][2], detection_param[2][3]
            x_min, y_min, x_max, y_max = convertBack(float(x), float(y), float(w), float(h))
            if x_min < 0:
                x_min = 0
            if y_min < 0:
                y_min = 0
            detection_coo_list.append(
                {"detect_label": zh_en_dir[detection_param[0].decode()], "pt1": (x_min, y_min), "pt2": (x_max, y_max)})
    return {"detection_coo_list": detection_coo_list, "img": img}


def compute_detection(detections, img, confidence_threshold):
    detection_coo_list = []
    for detection_param in detections:
        if detection_param[1] > confidence_threshold:
            x, y, w, h = detection_param[2][0], detection_param[2][1], detection_param[2][2], detection_param[2][3]
            x_min, y_min, x_max, y_max = convertBack(float(x), float(y), float(w), float(h))
            if x_min < 0:
                x_min = 0
            if y_min < 0:
                y_min = 0
            detection_coo_list.append(
                {"detect_label": detection_param[0].decode(), "pt1": (x_min, y_min), "pt2": (x_max, y_max)})
    return {"detection_coo_list": detection_coo_list, "img": img}


def normalize_detect_param(detections, confidence_limit):
    detection_coo_list = []
    for detection_param in detections:
        if detection_param[1] > confidence_limit:
            x, y, w, h = detection_param[2][0], detection_param[2][1], detection_param[2][2], detection_param[2][3]
            x_min, y_min, x_max, y_max = convertBack(float(x), float(y), float(w), float(h))
            if x_min < 0:
                x_min = 0
            if y_min < 0:
                y_min = 0
            detection_coo_list.append(
                {"object_name": detection_param[0].decode(), "object_confidence": detection_param[1],
                 "lt_box": {"lt_x": x_min, "lt_y": y_min}, "rl_box": {"rl_x": x_max, "rl_y": y_max}})
    return detection_coo_list


def detection_to_tracker(detections, confidence, zh_en_dir):
    """
    将初始识别参数转为跟踪坐标参数
    :return:
    """
    label_names = []
    label_boxes = []
    if detections:
        for detection_param in detections:
            if detection_param[1] > confidence:
                label_names.append(zh_en_dir[detection_param[0].decode()])
                label_boxes.append(
                    (detection_param[2][0], detection_param[2][1], detection_param[2][2], detection_param[2][3]))
    return label_boxes, label_names


if __name__ == '__main__':
    """
    {'detections': 
    [(b'dog', 0.9978259205818176, (221.85183715820312, 383.36724853515625, 196.34954833984375, 319.6354675292969)), 
     (b'bicycle', 0.9898183345794678, (343.392578125, 278.48504638671875, 451.8406677246094, 308.537109375)), 
     (b'truck', 0.9373040795326233, (582.4103393554688, 126.84490966796875, 217.21414184570312, 78.67939758300781))], 
    'img':
    }
    """
    mes = load_model(darknet_path="/home/hadoop/Documents/darknet-master-1/darknet-master",
                     configPath="/home/hadoop/Documents/darknet-master-1/darknet-master/cfg/yolov3.cfg",
                     weightPath="/home/hadoop/Documents/darknet-master-1/darknet-master/backup/yolov3.weights",
                     metaPath="/home/hadoop/Documents/darknet-master-1/darknet-master/cfg/coco.data")
    print(detection(darknet_path=mes[3],
                    net_main=mes[0],
                    meta_main=mes[1],
                    thresh=mes[2],
                    image_path="/home/hadoop/Documents/darknet-master-1/darknet-master/data/dog.jpg"))

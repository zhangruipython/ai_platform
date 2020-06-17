# -*- coding: utf-8 -*-
"""
@Time    : 2019-10-24 17:53
@Author  : zhangrui
@FileName: opencv_darknet_demo.py
@Software: PyCharm
使用opencv上的darknet
"""
import cv2
import numpy as np


# 获取输出层的名称
def getOutputsNames(net_work):
    # Get the names of all the layers in the network
    layersNames = net_work.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net_work.getUnconnectedOutLayers()]


# 处理网络输出层
def post_process(img, net_outs, classes):
    imgHeight = img.shape[0]
    imgWidth = img.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in net_outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * imgWidth)
                center_y = int(detection[1] * imgHeight)
                width = int(detection[2] * imgWidth)
                height = int(detection[3] * imgHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    detection_coo_list = []
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        label_confidence = '%.2f' % confidences[i]
        label_name = classes[classIds[i]]
        detection_coo_list.append({"extract_label": label_name, "pt1": (top, width),
                                   "pt2": ((left + width), (top + height))})
    return detection_coo_list


# 初始化参数
confThreshold = 0.5
nmsThreshold = 0.4
inpWidth = 416
inpHeight = 416


def load_model():
    classesFile = "D:/Documents/darknet/coco.names"
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')
    # Give the configuration and weight files for the model and load the network using them.
    modelConfiguration = "D:/Documents/darknet/yolov3.cfg"
    modelWeights = "D:/Documents/darknet/yolov3.weights"
    # 使用CPU
    cpu_net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    cpu_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    cpu_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return cpu_net, classes


if __name__ == '__main__':
    net = load_model()[0]
    my_classes = load_model()[1]
    img_path_list = ["D:/Documents/darknet/dog.jpg"]
    for img_path in img_path_list:
        frame = cv2.imread(img_path)
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        outs = net.forward(getOutputsNames(net))
        print(post_process(img=frame, net_outs=outs, classes=my_classes))
        t, _ = net.getPerfProfile()
        labels = '运行时间: %.2f 毫秒' % (t * 1000.0 / cv2.getTickFrequency())
        print(labels)

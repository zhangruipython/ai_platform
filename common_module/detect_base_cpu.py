# -*- coding: utf-8 -*-
"""
@Time    : 2020-06-19 14:45
@Author  : zhangrui
@FileName: detect_base_cpu.py
@Software: PyCharm
单张图片识别耗时2s
"""
import cv2
import numpy as np


class DetectBaseOpenCv:
    def __init__(self, classes_file, model_configuration, model_path, img_path):
        self.classes_file = classes_file
        self.model_configuration = model_configuration
        self.model_path = model_path
        self.img_path = img_path
        self.confThreshold = 0.5
        self.nmsThreshold = 0.4
        self.inpWidth, self.inpHeight = 416, 416

    @staticmethod
    def getOutputsNames(net_work):
        # Get the names of all the layers in the network
        layersNames = net_work.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i[0] - 1] for i in net_work.getUnconnectedOutLayers()]

    def load_model(self):
        with open(self.classes_file, "rt") as file:
            classes = file.read().rstrip('\n').split('\n')
        cpu_net = cv2.dnn.readNetFromDarknet(self.model_configuration, self.model_path)
        cpu_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        cpu_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return cpu_net, classes

    def post_process(self, img, net_outs, classes):
        imgHeight = img.shape[0]
        imgWidth = img.shape[1]
        classIds = []
        confidences = []
        boxes = []
        for out in net_outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > self.confThreshold:
                    center_x = detection[0] * imgWidth
                    center_y = detection[1] * imgHeight
                    width = detection[2] * imgWidth
                    height = detection[3] * imgHeight
                    # left = center_x - width / 2
                    # top = center_y - height / 2
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([center_x, center_y, width, height])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        detection_coo_list = []
        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            width = box[2]
            height = box[3]
            label_confidence = '%.2f' % confidences[i]
            label_name = classes[classIds[i]]
            # detection_coo_list.append({"extract_label": label_name, "pt1": (top, width),
            #                            "pt2": ((left + width), (top + height))})
            detection_coo_list.append((label_name.encode(), label_confidence, (x, y, width, height)))
        return {'detections': detection_coo_list, 'img': img}

    def make_detect(self):
        frame = cv2.imread(self.img_path)
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (self.inpWidth, self.inpHeight), [0, 0, 0], 1, crop=False)
        net, classes = self.load_model()
        net.setInput(blob)
        outs = net.forward(self.getOutputsNames(net))
        return self.post_process(frame, outs, classes)


if __name__ == '__main__':
    """
    x,y,w,h
    {'detections': 
    [(b'dog', 0.9978259205818176, (221.85183715820312, 383.36724853515625, 196.34954833984375, 319.6354675292969)), 
     (b'bicycle', 0.9898183345794678, (343.392578125, 278.48504638671875, 451.8406677246094, 308.537109375)), 
     (b'truck', 0.9373040795326233, (582.4103393554688, 126.84490966796875, 217.21414184570312, 78.67939758300781))], 
    'img':
    }
    """
    detect_base = DetectBaseOpenCv(classes_file="D:/Documents/darknet/coco.names",
                                   model_path="D:/Documents/darknet/yolov3.weights",
                                   model_configuration="D:/Documents/darknet/yolov3.cfg",
                                   img_path="D:/Documents/darknet/dog.jpg")
    print(detect_base.make_detect())

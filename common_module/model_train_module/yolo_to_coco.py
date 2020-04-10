# -*- coding: utf-8 -*-
"""
@Time    : 2020-03-23 13:58
@Author  : zhangrui
@FileName: yolo_to_coco.py
@Software: PyCharm
yolo格式数据集转化为coco格式数据集
注意点：detectron2读取图片使用的方法是：
*************
from PIL import Image, ImageOps
image = ImageOps.exif_transpose(image)
该方法会读取图片exif信息来获取图像宽高数据
所以在这里也需要使用相同方法获取图像宽高
*************
"""
import glob
import json
from PIL import Image, ImageOps
# import cv2
import numpy as np
from fvcore.common.file_io import PathManager


class YoloConvertCoco:
    def __init__(self, yolo_labels_dir, label_class_file, save_coco_path):
        self.yolo_labels_dir = yolo_labels_dir
        self.label_class_file = label_class_file
        self.save_coco_path = save_coco_path

    def Convert(self):
        # 定义coco数据集输出格式
        data_set = {"categories": [], "annotations": [], "images": []}

        # 读取class
        with open(self.label_class_file, "r") as f:
            classes = f.read().strip().split()
        for i, cls in enumerate(classes, 1):
            data_set["categories"].append({"id": i, "name": cls, "supercategory": 'mark'})
        txt_files = glob.glob(self.yolo_labels_dir + "/" + "*.txt")

        # 读取yolo标注格式的TXT文件
        count = 0
        annotation_id = 0
        for txt_file in txt_files:
            image_path = txt_file.replace("txt", "jpg")
            with PathManager.open(image_path, "rb") as f:
                try:
                    image = ImageOps.exif_transpose(Image.open(f))
                    image = np.asarray(image)
                    W, H = image.shape[1], image.shape[0]
                    data_set["images"].append({"file_name": image_path, "id": count, "width": W, "height": H})
                    with open(txt_file, "r") as fr:
                        label_list = fr.readlines()
                        for label in label_list:
                            label = label.strip().split()
                            bbox_width, bbox_height = float(label[3]) * W, float(label[4]) * H
                            center_x, center_y = float(label[1]) * W, float(label[2]) * W
                            x1, y1, x2, y2 = int(center_x - (bbox_width / 2)), int(center_y - (bbox_height / 2)), int(
                                center_x + (
                                        bbox_width / 2)), int(center_y + (bbox_height / 2))
                            data_set["annotations"].append({"area": H * W, "bbox": [x1, y1, x2 - x1, y2 - y1],
                                                            "category_id": int(label[0]) + 1, "image_id": count,
                                                            "id": annotation_id,
                                                            "iscrowd": 0,
                                                            "segmentation": [[x1, y1, x2, y1, x2, y2, x1, y2]]})
                            annotation_id += 1
                    count += 1
                except Exception as exp:
                    print("except is {a},image path is {b}".format(a=exp, b=image_path))
        with open(self.save_coco_path, "w") as f:
            json.dump(data_set, f)


if __name__ == '__main__':
    labels_dir = "/home/detectron2/train_data/data"
    class_file = "/home/detectron2/train_data/mask_labels.txt"
    coco_file = "/home/detectron2/train_data/coco_data.json"
    yolo_convert_coco = YoloConvertCoco(yolo_labels_dir=labels_dir, label_class_file=class_file,
                                        save_coco_path=coco_file)
    yolo_convert_coco.Convert()

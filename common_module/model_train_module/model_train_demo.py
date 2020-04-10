# -*- coding: utf-8 -*-
"""
@Time    : 2020-03-19 16:18
@Author  : zhangrui
@FileName: model_train_demo.py
@Software: PyCharm
coco类型数据集进行模型训练
"""
import glob
import cv2
import os
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances
from detectron2.data.datasets import load_coco_json


class ModelTrain:
    def __init__(self, train_data_path, train_data_type):
        """
        数据训练
        :param train_data_path: 存放图片和标注文件的目录（如果是coco数据集，则默认图片名称与标注文件名称对应且两者位于同一目录）
        :param train_data_type: 图片标注数据格式
        """
        self.train_data_path = train_data_path
        self.train_data_type = train_data_type

    def get_all_label_path(self):
        """
        获取文件目录所有图像和标注文件的绝对路径
        :return:
        """
        label_url = self.train_data_path + "*.txt"
        return glob.glob(label_url)

    @staticmethod
    def parse_label_file(label_file):
        """
        解析图像标注文件
        :param label_file: 图像标注具体文件
        :return:
        """
        label_list = []
        with open(label_file, "r") as f:
            for line in f.readlines():
                fields = line.strip().split()
                # label_index, x_min, y_min, x_max, y_max = int(fields[0]), float(fields[1]), float(fields[2]), float(
                #     fields[3]), float(fields[4])
                label_index, x, y, w, h = int(fields[0]), float(fields[1]), float(fields[2]), float(fields[3]), float(
                    fields[4])
                label_dic = {"label_index": label_index, "x": x, "y": y, "w": w, "h": h}
                label_list.append(label_dic)
        return label_list

    def get_tl_dicts(self):
        labels = self.get_all_label_path()
        dataset_dicts = []
        for label in labels:
            image_path = label.split(".")[0] + ".jpg"
            image_label = {"path": image_path, "boxes": self.parse_label_file(label)}
            img_height, img_weight = cv2.imread(image_path).shape[:2]
            record = dict()
            record["file_name"] = image_path
            record["img_height"] = img_height
            record["img_weight"] = img_weight
            objs = []
            for box in image_label["boxes"]:
                obj = {
                    "bbox": [box["x"], box["y"], box["w"], box["h"]],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "category_id": 0,
                    "iscrowd": 0
                }
                objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
        return dataset_dicts

    def train_coco_data(self, coco_json):
        dataset_name = "mask_train_data"
        DatasetCatalog.register(dataset_name,
                                lambda: load_coco_json(json_file=coco_json, image_root=self.train_data_path))
        MetadataCatalog.get(dataset_name).set(json_file=coco_json, image_root=self.train_data_path,
                                              evaluator_type="coco", thing_classes=["rightmask"],
                                              thing_dataset_id_to_contiguous_id={1: 0})
        cfg = get_cfg()
        cfg.merge_from_file("/home/detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        cfg.DATASETS.TRAIN = (dataset_name,)
        cfg.DATASETS.TEST = (dataset_name,)
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.MODEL.WEIGHTS = "/home/detectron2/train_data/model_final_280758.pkl"
        cfg.SOLVER.IMS_PER_BATCH = 2
        cfg.SOLVER.BASE_LR = 0.01  # 学习率
        cfg.SOLVER.MAX_ITER = 300  # 最大迭代次数
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        print("模型存储路径" + cfg.OUTPUT_DIR)
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()  # 开始训练


def model_check():
    """
    训练模型结果检测
    :return:
    """
    pass


if __name__ == '__main__':
    train_data_path = "/home/detectron2/train_data/data/"
    model_train = ModelTrain(train_data_path=train_data_path, train_data_type="coco")
    model_train.train_coco_data(coco_json="/home/detectron2/train_data/coco_data.json")

# -*- coding: utf-8 -*-
"""
@Time    : 2019-11-26 14:00
@Author  : zhangrui
@FileName: model_load.py
@Software: PyCharm
模型预加载
"""
import sys
import configparser
import os

sys.path.append('../')
sys.path.append('../../')


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


def load(config_path, model_name):
    cf = configparser.ConfigParser()
    cf.read(config_path, encoding="utf-8")
    darknet_path = cf.get(model_name, "darknet_path")
    cfg_path = cf.get(model_name, "config_path")
    weight_path = cf.get(model_name, "weight_path")
    meta_path = cf.get(model_name, "meta_path")
    return load_model(darknet_path, cfg_path, weight_path, meta_path)

# -*- coding: utf-8 -*-
"""
@Time    : 2019-12-11 13:45
@Author  : zhangrui
@FileName: picture_mix.py
@Software: PyCharm
短视频拆分为帧拼接为长图
1、获取两张输入的图像中关键点和局部不变性描述符，这里使用 SIFT 算法完成
2、匹配两张输入图像的局部不变性描述符
3、使用 RANSAC 算法基于我们匹配的特征向量合成 Homography 矩阵
4、使用第三步得到的 Homography 矩阵对图像进行拼接
"""
import cv2
import numpy as np
import sys

sys.setrecursionlimit(100000)  # 更改python对递归次数的限制


class PictureMix:
    def __init__(self, original_img_width):
        """
        :param original_img_width: 原始图片宽度（不需要去除黑色区域）
        """
        self.original_img_width = original_img_width

    def trim(self, frame):
        """
        去除拼接图像的黑色区域
        :param frame:
        :return:
        """
        # crop top
        if not np.sum(frame[0]):
            return self.trim(frame[1:])
        # crop bottom
        elif not np.sum(frame[-1]):
            return self.trim(frame[:-2])
        # crop left
        elif not np.sum(frame[:, 0]):
            return self.trim(frame[:, -1:])
        # crop right
        elif not np.sum(frame[:, -1]):
            return self.trim(frame[:, :-2])
        return frame

    @staticmethod
    def mix(trim_part01, trim_part02):
        # 仅有x轴（类似宽度）left 开始重叠的最左端 right 为重叠区域最右边像素坐标
        MIN = 10  # 好点的最小数量标准
        """
        创建surf对象 设定Hessian Threshold阈值=500，阈值越大检测到的特征越少
        upright=True不检测关键点方向
        """
        surf = cv2.xfeatures2d.SURF_create(500, nOctaves=4, extended=False, upright=True)
        # 计算图像关键点和sift特征向量
        kp1, des01 = surf.detectAndCompute(trim_part01, None)
        kp2, des02 = surf.detectAndCompute(trim_part02, None)
        # 建立FLANN匹配器参数
        FLANN_INDEX_KDTREE = 0
        # 配置索引，密度树的数量为5
        indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        # 指定递归次数
        searchParams = dict(checks=50)
        # 使用特征匹配算法（最近邻搜索KNN）
        flann = cv2.FlannBasedMatcher(indexParams, searchParams)
        match = flann.knnMatch(des01, des02, k=2)
        good_point = [a[0] for a in match if a[0].distance < 0.75 * a[1].distance]
        if len(good_point) > MIN:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_point]).reshape(-1, 1, 2)
            ano_pts = np.float32([kp2[m.trainIdx].pt for m in good_point]).reshape(-1, 1, 2)
            # 传入图像A关键点坐标和图像B关键点坐标，使用随机抽样一致算法迭代，生成变化矩阵
            M, mask = cv2.findHomography(src_pts, ano_pts, cv2.RANSAC, 5.0)
            warpImg = cv2.warpPerspective(trim_part02, np.linalg.inv(M),
                                          (trim_part01.shape[1] + trim_part02.shape[1],
                                           trim_part02.shape[0]))
            warpImg[0:trim_part01.shape[0], 0:trim_part01.shape[1]] = trim_part01
            return warpImg
        else:
            print("not enough matches!")

    def photo_mix(self, img_part_list):
        """
        1、图像去除黑色区域
        2、图像拼接
        :param img_part_list:拼接图像list,list[0]为右侧图片，list[1]为左侧图片
        :return:拼接图片
        """
        if img_part_list[0].shape[1] == self.original_img_width:
            stitch_img = self.mix(trim_part01=img_part_list[1], trim_part02=img_part_list[0])
        else:
            # img_part01为图像左部分，img_part02为图像右部分
            img_part01, img_part02 = img_part_list[1], img_part_list[0]
            trim_part01, trim_part02 = img_part01, img_part02
            trim01_height, trim01_width = trim_part01.shape[:2]
            trim02_height, trim02_width = trim_part02.shape[:2]
            if trim01_width > trim02_width:
                trim_part01 = trim_part01[0:trim01_height, 0:trim02_width]
            elif trim01_width < trim02_width:
                trim_part02 = trim_part02[0:trim01_height, 0:trim01_width]
            stitch_img = self.trim(self.mix(trim_part01, trim_part02))
        # 图像拼接后在右侧会有小部分黑色区域
        return stitch_img[0:stitch_img.shape[0], 0:int(stitch_img.shape[1] * 0.95)]


if __name__ == '__main__':
    img01_path = "C:\\rongze\\data\\yike_picture\\demo05\\mix0_120.jpg"
    img02_path = "C:\\rongze\\data\\yike_picture\\demo05\\mix160_280.jpg"
    joint_img_path = "C:\\rongze\\data\\yike_picture\\demo05\\mix0_280.jpg"
    img01 = cv2.imread(img01_path)
    img02 = cv2.imread(img02_path)
    picture_mix = PictureMix(original_img_width=1280)
    mix_img = picture_mix.photo_mix(img_part_list=[img01, img02])
    # mix_img = picture_mix.trim(mix_img)
    cv2.imwrite(joint_img_path, mix_img)

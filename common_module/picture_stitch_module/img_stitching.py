# -*- coding: utf-8 -*-
"""
@Time    : 2019-12-11 15:03
@Author  : zhangrui
@FileName: img_stitching.py
@Software: PyCharm
"""
import cv2
import glob
import sys
sys.path.append("../../")
from ai_rpc_service.object_count_service.picture_stitch_module.picture_mix import PictureMix
from concurrent.futures import ProcessPoolExecutor


class MorePictureJoint:
    def __init__(self, video_path, split_len):
        self.split_len = split_len
        self.video_path = video_path

    def split_video(self):
        capture = cv2.VideoCapture(self.video_path)
        frame_index = 0
        frame_num = 0
        frame_img_list = []
        while capture.isOpened():
            success, frame = capture.read()
            if not success:
                break
            else:
                """
                保存2的指数张图片 16张
                """
                # if (frame_index - 1) % (self.split_len + 1) == 0 or frame_index % (self.split_len + 1) == 0:
                if frame_index % self.split_len == 0:
                    cv2.imwrite("C:/Users/hadoop/rongerai/property_check/img/" + str(frame_index) + ".jpg", frame)
                    frame_img_list.append(frame)
                    frame_num += 1
                elif frame_num == 8:
                    break
                frame_index += 1
        capture.release()
        return frame_img_list

    @staticmethod
    def group_two(layer_list):
        """
        将list中元素两两划分
        :param layer_list:传入list
        :return: 二维list
        """
        return [layer_list[i:i + 2] for i in range(0, len(layer_list), 2)]

    @staticmethod
    def load_frame():
        pic_list = glob.glob("C:/rongze/data/yike_picture/mix/*")
        return [cv2.imread(pic) for pic in pic_list]

    def stitch_img(self):
        # time_01 = time.time()
        img_list = self.split_video()
        # img_list = self.load_frame()
        # print("视频拆分所耗时{a}".format(a=time.time() - time_01))
        # print(len(img_list))
        # process_count = os.cpu_count()
        picture_mix = PictureMix(original_img_width=img_list[0].shape[1])
        while True:
            if len(img_list) == 1:
                # 如果img数量为1则结束循环
                break
            elif len(img_list) % 2 == 0:
                # 如果img数量为偶数，则两两划分数据进行分组图像合并，最后生成两两合并后的img_list
                group_list = self.group_two(img_list)
                # r_mix = list(ProcessPoolExecutor(max_workers=process_count).map(picture_mix.photo_mix, group_list))
                r_mix = list(map(picture_mix.photo_mix, group_list))
                img_list = r_mix
            else:
                # 如果img数量为奇数，则先排除最后一个元素，对剩余元素进行两两分组图像合并，最后生成的img_list中再加入最后一个元素
                group_list = self.group_two(img_list[:-1])
                r_mix = list(ProcessPoolExecutor(max_workers=len(group_list)).map(picture_mix.photo_mix, group_list))
                # r_mix = list(map(picture_mix.photo_mix, group_list))
                r_mix.append(img_list[-1])
                img_list = r_mix
        return img_list[0]


if __name__ == '__main__':
    # time_start = time.time()
    picture_stitch = MorePictureJoint(video_path="C:/rongze/data/yike_picture/demo04.mp4", split_len=40)
    stitching_img = picture_stitch.stitch_img()
    # a = picture_stitch.split_video()
    # img = trim(stitching_img)
    # print("图像拼接时长{a}".format(a=time.time() - time_start))
    cv2.imwrite("C:\\rongze\\data\\yike_picture\\demo05\\demo_mix01.jpg", stitching_img)

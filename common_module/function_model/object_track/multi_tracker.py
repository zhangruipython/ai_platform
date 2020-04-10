# -*- coding: utf-8 -*-
"""
@Time    : 2020-04-09 14:41
@Author  : zhangrui
@FileName: multi_tracker.py
@Software: PyCharm
多目标跟踪
"""
import cv2
from random import randint

trackTypes = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']


def create_tracker(track_type):
    if track_type == trackTypes[0]:
        tracker = cv2.TrackerBoosting_create()
    elif track_type == trackTypes[1]:
        tracker = cv2.TrackerMIL_create()
    elif track_type == trackTypes[2]:
        tracker = cv2.TrackerKCF_create()
    elif track_type == trackTypes[3]:
        tracker = cv2.TrackerTLD_create()
    elif track_type == trackTypes[4]:
        tracker = cv2.TrackerMedianFlow_create()
    elif track_type == trackTypes[5]:
        tracker = cv2.TrackerGOTURN_create()
    elif track_type == trackTypes[6]:
        tracker = cv2.TrackerMOSSE_create()
    elif track_type == trackTypes[7]:
        tracker = cv2.TrackerCSRT_create()
    else:
        tracker = None
    return tracker


if __name__ == '__main__':
    trackType = "MEDIANFLOW"
    # 读取视频流
    video_url = 0+cv2.CAP_DSHOW
    # video_url = "D:/rongze/picture/car/big_front_car.mp4"
    capture = cv2.VideoCapture(video_url)
    success, first_frame = capture.read()
    boxes = []
    colors = []
    while True:
        bbox = cv2.selectROI("MultiTrackers", first_frame)
        boxes.append(bbox)
        colors.append((randint(64, 255), randint(64, 255), randint(64, 255)))
        k = cv2.waitKey(0) & 0xFF

        if k == ord("q"):  # 按下q退出
            capture.release()
            cv2.destroyAllWindows()
            break

    print("所选择的边框坐标{}".format(boxes))
    multiTracker = cv2.MultiTracker_create()
    # 初始化多个跟踪器
    for box in boxes:
        multiTracker.add(create_tracker(trackType), first_frame, box)
    capture01 = cv2.VideoCapture(video_url)
    while capture01.isOpened():
        success, frame = capture01.read()
        if success:
            print("读取成功")
            success, boxes = multiTracker.update(frame)
            print(success, boxes)
            # 绘制跟踪框
            for i, new_box in enumerate(boxes):
                p1 = (int(new_box[0]), int(new_box[1]))
                p2 = (int(new_box[0] + new_box[2]), int(new_box[1] + new_box[3]))
                cv2.rectangle(frame, p1, p2, colors[i], 2, 1)
            cv2.imshow("MultiTracker", frame)
            k = cv2.waitKey(1)
            if k == ord("q"):
                break
        else:
            capture.release()
            break

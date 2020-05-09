# -*- coding: utf-8 -*-
"""
@Time    : 2020-04-08 13:36
@Author  : zhangrui
@FileName: detect_streamlit.py
@Software: PyCharm
"""
import base64
import time
import glob
import json
import requests
import streamlit as st
import numpy as np
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


def make_track():
    trackType = "CSRT"
    # 读取视频流(添加cv2.CAP_DSHOW解决windows打开前置摄像头报错问题)
    # video_url = 0 + cv2.CAP_DSHOW
    video_url = 0
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
                capture.release()
                cv2.destroyAllWindows()
                break
        else:
            capture.release()
            cv2.destroyAllWindows()
            break


# Streamlit encourages well-structured code, like starting execution in a main() function.
def main():
    # 渲染Markdown文件
    readme_text = st.markdown(get_file_content_as_string("introduce.md"))
    st.sidebar.title("执行操作")
    app_mode = st.sidebar.selectbox("选择执行模块",
                                    ["各模块介绍", "人体关键点检测服务", "全景分割服务", "口罩佩戴检测", "实例分割服务", "服装检测", "目标跟踪", "查看源码"])
    if app_mode == "各模块介绍":
        pass
        # st.sidebar.success('To continue select "Run the app".')
    elif app_mode == "人体关键点检测服务":
        readme_text.empty()
        st.markdown("# 进行人体关键点检测")
        img_list = glob.glob("image/*.jpg")
        print(list(map(show_img, img_list)))
        # object_type = st.sidebar.selectbox("请选择图片", ["0.jpg", "1.jpg", "2.jpg", "3.jpg"])
        object_type = st.sidebar.selectbox("请选择图片", img_list)
        if st.sidebar.button("开始检测", key=None):
            # picture_path = "image/" + object_type
            # picture_path = object_type
            show_image(img_path=object_type)  # 图片展示
            '开始识别'
            mat_img = client(server_url="http://192.168.1.54:5001/object_detect/key_points", image_path=object_type)
            # Add a placeholder
            latest_iteration = st.empty()
            bar = st.progress(0)

            for i in range(20):
                # Update the progress bar with each iteration.
                latest_iteration.text(f'Iteration {i+1}')
                bar.progress(i + 1)
                time.sleep(0.1)

            '...识别结束!'
            st.image(mat_img)

    elif app_mode == "全景分割服务":
        readme_text.empty()
        st.markdown("# 全景分割服务")
        img_list = glob.glob("image/*.jpg")
        print(list(map(show_img, img_list)))
        object_type = st.sidebar.selectbox("请选择图片", img_list)
        if st.sidebar.button("开始检测", key=None):
            # picture_path = "image/" + object_type
            # picture_path = object_type
            show_image(img_path=object_type)  # 图片展示
            '开始识别'
            mat_img = client(server_url="http://192.168.1.54:5001/object_detect/panoramic_segm/",
                             image_path=object_type)
            # Add a placeholder
            latest_iteration = st.empty()
            bar = st.progress(0)
            for i in range(20):
                # Update the progress bar with each iteration.
                latest_iteration.text(f'Iteration {i+1}')
                bar.progress(i + 1)
                time.sleep(0.1)

            '...识别结束!'
            st.image(mat_img)
    elif app_mode == "服装检测":
        readme_text.empty()
        st.markdown("# 服装检测")
        img_list = glob.glob("image/*.jpg")
        print(list(map(show_img, img_list)))
        object_type = st.sidebar.selectbox("请选择图片", img_list)
        if st.sidebar.button("开始检测", key=None):
            picture_path = object_type
            show_image(img_path=picture_path)  # 图片展示
            '开始识别'
            mat_img = client(server_url="http://192.168.1.54:5001/object_detect/cloth_detect/", image_path=picture_path)
            # Add a placeholder
            latest_iteration = st.empty()
            bar = st.progress(0)

            for i in range(20):
                # Update the progress bar with each iteration.
                latest_iteration.text(f'Iteration {i+1}')
                bar.progress(i + 1)
                time.sleep(0.1)

            '...识别结束!'
            st.image(mat_img)
    elif app_mode == "口罩佩戴检测":
        readme_text.empty()
        st.markdown("# 口罩佩戴检测")
        img_list = glob.glob("image/*.jpg")
        print(list(map(show_img, img_list)))
        object_type = st.sidebar.selectbox("请选择图片", img_list)
        if st.sidebar.button("开始检测", key=None):
            picture_path = object_type
            show_image(img_path=picture_path)  # 图片展示
            '开始识别'
            mat_img = client(server_url="http://192.168.1.54:5001/object_detect/mask_detect/", image_path=picture_path)
            # Add a placeholder
            latest_iteration = st.empty()
            bar = st.progress(0)

            for i in range(20):
                # Update the progress bar with each iteration.
                latest_iteration.text(f'Iteration {i+1}')
                bar.progress(i + 1)
                time.sleep(0.1)

            '...识别结束!'
            st.image(mat_img)
    elif app_mode == "目标跟踪":
        readme_text.empty()
        st.markdown("# 目标跟踪")
        # make_track()
        st.markdown("## 按下'q'结束目标跟踪")
        if st.button("开始目标跟踪", key=None):
            make_track()

    elif app_mode == "查看源码":
        readme_text.empty()
        st.code(get_file_content_as_string("detect_streamlit.py"))
    elif app_mode == "实例分割服务":
        readme_text.empty()
        st.markdown("# 实例分割服务")
        img_list = glob.glob("image/*.jpg")
        print(list(map(show_img, img_list)))
        object_type = st.sidebar.selectbox("请选择图片", img_list)
        if st.sidebar.button("开始检测", key=None):
            picture_path = object_type
            show_image(img_path=picture_path)  # 图片展示
            '开始识别'
            mat_img = client(server_url="http://192.168.1.54:5001/object_detect/instance_segm/",
                             image_path=picture_path)
            # Add a placeholder
            latest_iteration = st.empty()
            bar = st.progress(0)

            for i in range(20):
                # Update the progress bar with each iteration.
                latest_iteration.text(f'Iteration {i+1}')
                bar.progress(i + 1)
                time.sleep(0.1)

            '...识别结束!'
            st.image(mat_img)


def get_file_content_as_string(path):
    with open(path, "rb") as f:
        return f.read().decode("utf-8")


def show_img(img_path):
    img_mat = cv2.imread(img_path)
    # BGR->RGB
    st.sidebar.image(use_column_width=True, image=img_mat[:, :, [2, 1, 0]], caption=img_path.split("image/")[1])
    return 1


def show_image(img_path):
    img_mat = cv2.imread(img_path)
    # BGR->RGB
    st.image(use_column_width=False, image=img_mat[:, :, [2, 1, 0]], caption=img_path.split("image/")[1])


@st.cache()
def client(server_url, image_path):
    with open(image_path, "rb") as f:
        base64_data = base64.b64encode(f.read())
        s = base64_data.decode()
    response = requests.post(url=server_url, json={"img_base64": "data:image/jpg;base64," + str(s)})
    out_img_base64 = json.loads(response.text)["base64_str"]
    img_str = base64.b64decode(out_img_base64)
    np_arr = np.frombuffer(img_str, np.uint8)
    mat_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)[:, :, ::-1]
    return mat_img


def frame_selector_ui():
    """
    侧边栏UI
    :return:
    """
    pass


if __name__ == "__main__":
    main()

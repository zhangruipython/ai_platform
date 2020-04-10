# -*- coding: utf-8 -*-
"""
@Time    : 2020-04-08 13:36
@Author  : zhangrui
@FileName: detect_streamlit.py
@Software: PyCharm
"""
import base64
import time

import cv2
import json
import requests
import streamlit as st
import numpy as np


# Streamlit encourages well-structured code, like starting execution in a main() function.
def main():
    # 渲染Markdown文件
    readme_text = st.markdown(get_file_content_as_string("introduce.md"))

    # Download external dependencies.
    # 下载依赖文件
    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title("执行操作")
    app_mode = st.sidebar.selectbox("选择执行模块",
                                    ["各模块介绍", "人体关键点检测服务", "全景分割服务", "口罩佩戴检测", "实例分割服务", "查看源码"])
    if app_mode == "各模块介绍":
        pass
        # st.sidebar.success('To continue select "Run the app".')
    elif app_mode == "人体关键点检测服务":
        readme_text.empty()
        st.markdown("# 进行人体关键点检测")
        show_img("image/0.jpg")
        show_img("image/1.jpg")
        show_img("image/2.jpg")
        show_img("image/3.jpg")
        object_type = st.sidebar.selectbox("请选择图片", ["0.jpg", "1.jpg", "2.jpg", "3.jpg"])
        if st.sidebar.button("开始检测", key=None):
            picture_path = "image/" + object_type
            show_image(img_path=picture_path)  # 图片展示
            '开始识别'
            mat_img = client(server_url="http://192.168.1.54:5001/object_detect/key_points", image_path=picture_path)
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

        show_img("image/0.jpg")
        show_img("image/1.jpg")
        show_img("image/2.jpg")
        show_img("image/3.jpg")
        object_type = st.sidebar.selectbox("请选择图片", ["0.jpg", "1.jpg", "2.jpg", "3.jpg"])
        if st.sidebar.button("开始检测", key=None):
            picture_path = "image/" + object_type
            show_image(img_path=picture_path)  # 图片展示
            '开始识别'
            mat_img = client(server_url="http://192.168.1.54:5001/object_detect/panoramic_segm/", image_path=picture_path)
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

        show_img("image/0.jpg")
        show_img("image/1.jpg")
        show_img("image/2.jpg")
        show_img("image/3.jpg")
        object_type = st.sidebar.selectbox("请选择图片", ["0.jpg", "1.jpg", "2.jpg", "3.jpg"])
        if st.sidebar.button("开始检测", key=None):
            picture_path = "image/" + object_type
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

    elif app_mode == "查看源码":
        readme_text.empty()
        st.code(get_file_content_as_string("detect_streamlit.py"))
    elif app_mode == "实例分割服务":
        readme_text.empty()
        st.markdown("# 实例分割服务")

        show_img("image/0.jpg")
        show_img("image/1.jpg")
        show_img("image/2.jpg")
        show_img("image/3.jpg")
        object_type = st.sidebar.selectbox("请选择图片", ["0.jpg", "1.jpg", "2.jpg", "3.jpg"])
        if st.sidebar.button("开始检测", key=None):
            picture_path = "image/" + object_type
            show_image(img_path=picture_path)  # 图片展示
            '开始识别'
            mat_img = client(server_url="http://192.168.1.54:5001/object_detect/instance_segm/", image_path=picture_path)
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

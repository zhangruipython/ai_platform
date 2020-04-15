# ai_rpc_service
基于yolov3目标检测和detectron2视觉平台搭建web服务和rpc服务
## Requirement
- Python3.5
- OpenCV
- Detctron2
- YOLOV3
- Flask
- Jupyter Notebook
- Streamlit
---------
## 服务启动
**在ai_rpc_service/object_detect_service/web_client目录下执行命令：**
**gunicorn -w 3 -b 0.0.0.0:port flaskr_web:app > server.log 2>&1 &**
## jupyter notebook文件目录
**ai_rpc_service/show_platform/picture_detect.ipynb**
---------
## 可视化交互界面如下
![image](https://i.loli.net/2020/04/08/PJ1NklYRCfq9bn5.png)
---------
## 添加opencv contrib模块中多目标跟踪算法模块，方法调用路径：
**common_module/function_model/object_track/multi_tracker.py**
# -*- coding: utf-8 -*-
"""
@Time    : 2020-03-02 12:12
@Author  : zhangrui
@FileName: flaskr_web.py
@Software: PyCharm
python获取本机IP
"""
import socket
host_name = socket.gethostname()
ip = socket.gethostbyname(host_name)
print(f"{ip}{host_name}")

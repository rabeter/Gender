#!/usr/bin/env python
"""
从摄像头中获取图像实时监测
"""
import numpy as np
import cv2
from GenderTrain import Model


def detect(img, cascade):
    """
    检测图像是否含有人脸部分
    :param img: 待检测帧图像
    :param cascade: 面部对象检测器
    :return: 面部图像标记
    """
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects


def draw_rects(img, rects, color):
    """
    根据图像标记人脸区域与性别
    :param img: 
    :param rects: 
    :param color: 
    :return: 
    """
    for x, y, w, h in rects:
        face = img[x:x+w,y:y+h]
        face = cv2.resize(face,(224,224))
        if gender.predict(face)==1:
            text = "Male"
        else:
            text = "Female"
        cv2.rectangle(img, (x, y), (w, h), color, 2)
        cv2.putText(img, text, (x, h), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), lineType=cv2.LINE_AA)


if __name__ == '__main__':
    cascade = cv2.CascadeClassifier( "haarcascade_frontalface_default.xml")
    cam = cv2.VideoCapture(0)
    # 获取摄像头视频
    gender = Model()
    gender.load()
    # 加载性别模型
    while True:
        ret, img = cam.read()
        # 读取帧图像
        rects = detect(img, cascade)
        vis = img.copy()
        draw_rects(vis, rects, (0, 255, 0))
        cv2.imshow('Gender', vis)
        if cv2.waitKey(5) == 27:
            break
    cv2.destroyAllWindows()

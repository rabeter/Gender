#!/usr/bin/env python
"""
读取图像进行检测人脸性别
"""

import numpy as np
import cv2
from GenderTrain import Model



def draw_rects(img, rects):
    """
    根据图像标记人脸区域与性别
    :param img: 
    :param rects: 
    :return: 
    """
    for x, y, w, h in rects:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 00), 2)
        face = img
        face = cv2.resize(face,(224,224))
        if Gender.predict(face)==1:
            text = "Male"
        else:
            text = "Female"
        cv2.putText(img, text, (x, h), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
Gender = Model()
Gender.load()

img = cv2.imread('../../img/2991300_1982-04-03_2014.jpg')
faces = face_cascade.detectMultiScale(img, 1.3, 5)
draw_rects(img,faces)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
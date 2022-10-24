#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# @Author : Bismarckkk
# @Site   : https://github.com/bismarckkk
# @File   : cameraShot.py

# Quick Start
# 1. Modify global variable path.
# 2. Modify the topic name at L52 to your camera node's topic.
# 3. Launch roscore and your camera node.
# 4. Run this file. While pressing "Enter" key, one image will be saved.

# roscore
# rosrun mv_driver mv_driver_node
# cd /CLionProjects/RoboCup/TrainTools/notebooks
# python3 shot.py

import logging
import sys
from queue import Queue
from threading import Thread

import cv2
import numpy as np
import rospy
from rc_msgs.msg import raw_img

i = 0
path = '/home/stevegao/CLionProjects/RoboCup/TrainTools/shot/nn_%i.jpg'
queue = Queue(maxsize=2)


def img_to_cv2(_img_msg: raw_img, _):
    img_msg = _img_msg.color
    dtype = np.dtype("uint8")
    dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
    image_opencv = np.ndarray(shape=(img_msg.height, img_msg.width, 3),
                              dtype=dtype, buffer=img_msg.data)
    if img_msg.is_bigendian == (sys.byteorder == 'little'):
        image_opencv = image_opencv.byteswap().newbyteorder()
    if img_msg.encoding == 'bgr8':
        pass
    elif img_msg.encoding == 'rgb8':
        image_opencv = image_opencv[:, :, [2, 1, 0]]
    if queue.full():
        queue.get()
    queue.put(image_opencv)


def fixLogging(level=logging.WARNING):
    console = logging.StreamHandler()
    console.setLevel(level)
    logging.getLogger('').addHandler(console)
    formatter = logging.Formatter('%(levelname)-8s:%(name)-12s: %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


if __name__ == '__main__':
    rospy.init_node('shot')
    sub = rospy.Subscriber('/raw_img', raw_img, img_to_cv2, 1)
    thread = Thread(target=rospy.spin, name='ros', daemon=True)
    thread.start()
    fixLogging(logging.INFO)
    while True:
        input()
        if queue.empty():
            logging.info("Error: couldn't get image")
        else:
            cv2.imwrite(path % i, queue.get())
            logging.info(path % i + ' write')
            i += 1

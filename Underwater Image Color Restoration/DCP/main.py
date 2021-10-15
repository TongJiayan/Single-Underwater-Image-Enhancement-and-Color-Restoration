from GuidedFilter import GuidedFilter
from evaluation import getScore

__author__ = 'wy'
import datetime
import os
import math
import numpy as np
import cv2
import os
import numpy as np
import cv2
import natsort
import xlwt
import argparse
import time


# 用于排序时存储原来像素点位置的数据结构
class Node(object):
    def __init__(self, x, y, value):
        self.x = x
        self.y = y
        self.value = value

    def printInfo(self):
        print(self.x, self.y, self.value)


# 获取最小值矩阵
# 获取BGR三个通道的最小值
def getMinChannel(img):
    # 输入检查
    if len(img.shape) == 3 and img.shape[2] == 3:
        pass
    else:
        print("bad image shape, input must be color image")
        return None
    imgGray = img.min(axis=2)

    return imgGray


# 获取暗通道
def getDarkChannel(img, blockSize):
    # 输入检查
    if len(img.shape) == 2:
        pass
    else:
        print("bad image shape, input image must be two demensions")
        return None

    # blockSize检查
    if blockSize % 2 == 0 or blockSize < 3:
        print('blockSize is not odd or too small')
        return None
    
    # 计算addSize
    addSize = int((blockSize - 1) / 2)
    newHeight = img.shape[0] + blockSize - 1
    newWidth = img.shape[1] + blockSize - 1

    # 中间结果
    imgMiddle = np.zeros((newHeight, newWidth))
    imgMiddle[:, :] = 255
    imgMiddle[addSize:newHeight - addSize, addSize:newWidth - addSize] = img
    imgDark = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    
    # 计算每个block中最小值
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            imgDark[i,j] = imgMiddle[i:i+blockSize, j:j+blockSize].min()
   
    return imgDark

# 获取全局大气光强度
def getAtomsphericLight(darkChannel, img, meanMode=False, percent=0.001):
    size = darkChannel.shape[0] * darkChannel.shape[1]
    height = darkChannel.shape[0]
    width = darkChannel.shape[1]

    nodes = []

    # 用一个链表结构(list)存储数据
    for i in range(0, height):
        for j in range(0, width):
            oneNode = Node(i, j, darkChannel[i, j])
            nodes.append(oneNode)

    # 排序
    nodes = sorted(nodes, key=lambda node: node.value, reverse=True)

    atomsphericLight = 0

    # 原图像像素过少时，只考虑第一个像素点
    if int(percent * size) == 0:
        for i in range(0, 3):
            if img[nodes[0].x, nodes[0].y, i] > atomsphericLight:
                atomsphericLight = img[nodes[0].x, nodes[0].y, i]
        return atomsphericLight

    # 开启均值模式
    if meanMode:
        sum = 0
        for i in range(0, int(percent * size)):
            for j in range(0, 3):
                sum = sum + img[nodes[i].x, nodes[i].y, j]

        atomsphericLight = int(sum / (int(percent * size) * 3))
        return atomsphericLight

    # 获取暗通道前0.1%(percent)的位置的像素点在原图像中的最高亮度值
    for i in range(0, int(percent * size)):
        for j in range(0, 3):
            if img[nodes[i].x, nodes[i].y, j] > atomsphericLight:
                atomsphericLight = img[nodes[i].x, nodes[i].y, j]

    return atomsphericLight


# 恢复原图像
# Omega 去雾比例 参数
# t0 最小透射率值
def getRecoverScene(img, omega=0.95, t0=0.1, blockSize=15, meanMode=False, percent=0.001):
    
    gimfiltR = 50  # 引导滤波时半径的大小
    eps = 10 ** -3  # 引导滤波时epsilon的值
    
    print("Get MinChannel...")
    imgGray = getMinChannel(img)
    print("Get DarkChannel...")
    imgDark = getDarkChannel(imgGray, blockSize=blockSize)
    print("Get AtomsphericLight...")
    atomsphericLight = getAtomsphericLight(imgDark, img, meanMode=meanMode, percent=percent)

    imgDark = np.float64(imgDark)
    transmission = 1 - omega * imgDark / atomsphericLight

    print("Get Transmision...")
    guided_filter = GuidedFilter(img, gimfiltR, eps)
    transmission = guided_filter.filter(transmission)
    
    # 防止出现t小于0的情况
    # 对t限制最小值为0.1

    transmission = np.clip(transmission, t0, 0.9)

    print("Get sceneRadiance...")
    sceneRadiance = np.zeros(img.shape)
    for i in range(0, 3):
        img = np.float64(img)
        sceneRadiance[:, :, i] = (img[:, :, i] - atomsphericLight) / transmission + atomsphericLight

    sceneRadiance = np.clip(sceneRadiance, 0, 255)
    sceneRadiance = np.uint8(sceneRadiance)

    return transmission,sceneRadiance


def run(args):
    root_path = "/home/tongjiayan/UnderwaterImageRestoration/densedepth_based_restoration/datasets"
    dataset_path = os.path.join(root_path, args.dataset)
    if args.score != 'None':
        score = open(args.score, mode = 'a',encoding='utf-8')
    np.seterr(over='raise') # 触发FloatingPointError
    time_start = time.time()
    for image_name in os.listdir(dataset_path):
        image_path = os.path.join(dataset_path, image_name)
        if os.path.isfile(image_path):
            print('********    file   ********',image_name)
            img = cv2.imread(image_path) # BGR格式
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if args.score != 'None':
                print("Get score...")
                UICM, UISM, UIConM, UIQM, UCIQE = getScore(np.array(img))
                print(image_name, file=score)
                print("{0}\t{1}\t{2}\t{3}\t{4}".format(UICM, UISM, UIConM, UIQM, UCIQE), file=score)
                print("Finished to get score!")
            try:
                transmission, sceneRadiance = getRecoverScene(img)
            except ValueError:
                print("Fail to recover " + image_name + "!")
            else:
                save_path = os.path.join("OutputImages", args.dataset)
                output_image = image_name.split('.')[0] + '_DCP.jpg'
                output_depth = image_name.split('.')[0] + '_DCP_TM.jpg'
                print(os.path.join(save_path, output_depth))
                cv2.imwrite(os.path.join(save_path, output_depth), np.uint8(transmission * 255))
                if args.score != 'None':
                    UICM, UISM, UIConM, UIQM, UCIQE = getScore(np.array(sceneRadiance))
                    print("{0}\t{1}\t{2}\t{3}\t{4}".format(UICM, UISM, UIConM, UIQM, UCIQE), file=score)
                sceneRadiance = cv2.cvtColor(sceneRadiance, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(save_path, output_image), sceneRadiance)
                print("Done!")
    
    time_end = time.time()
    if args.score != 'None':
        print("TIME COST = {0}".format(time_end-time_start), file=score)
        score.close()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--score', default='None', help='score log')
    parser.add_argument('--dataset', default='RUCB/A3-05', help='Input image directory')
    args = parser.parse_args()
    run(args)

    


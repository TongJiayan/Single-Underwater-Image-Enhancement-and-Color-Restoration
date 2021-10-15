import os

import datetime
import numpy as np
import cv2
import natsort
import argparse
import time

from GuidedFilter import GuidedFilter
from backgroundLight import BLEstimation
from depthMapEstimation import depthMap
from depthMin import minDepth
from getRGBTransmission import getRGBTransmissionESt
from global_Stretching import global_stretching
from refinedTransmissionMap import refinedtransmissionMap

from sceneRadiance import sceneRadianceRGB
from evaluation import getScore

def getRecoverScene(img):
    gimfiltR = 50  # 引导滤波时半径的大小
    eps = 10 ** -3  # 引导滤波时epsilon的值

    DepthMap = depthMap(img)
    DepthMap = global_stretching(DepthMap)
    guided_filter = GuidedFilter(img, gimfiltR, eps)
    refineDR = guided_filter.filter(DepthMap)
    refineDR = np.clip(refineDR, 0,1)

    AtomsphericLight = BLEstimation(img, DepthMap) * 255

    d_0 = minDepth(img, AtomsphericLight)
    d_f = 8 * (DepthMap + d_0)
    transmissionB, transmissionG, transmissionR = getRGBTransmissionESt(d_f)

    transmission = refinedtransmissionMap(transmissionB, transmissionG, transmissionR, img)
    sceneRadiance = sceneRadianceRGB(img, transmission, AtomsphericLight)

    return sceneRadiance

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
            img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 转换成RGB格式
            if args.score != 'None':
                print("Get score...")
                UICM, UISM, UIConM, UIQM, UCIQE = getScore(np.array(img_RGB))
                print(image_name, file=score)
                print("{0}\t{1}\t{2}\t{3}\t{4}".format(UICM, UISM, UIConM, UIQM, UCIQE), file=score)
                print("Finished to get score!")
            try:
                sceneRadiance = getRecoverScene(img)
            except FloatingPointError:
                print("Fail to recover " + image_name + "!")
            else:
                if args.score != 'None':
                    sceneRadiance_RGB = cv2.cvtColor(sceneRadiance, cv2.COLOR_BGR2RGB)
                    UICM, UISM, UIConM, UIQM, UCIQE = getScore(np.array(sceneRadiance_RGB))
                    print("{0}\t{1}\t{2}\t{3}\t{4}".format(UICM, UISM, UIConM, UIQM, UCIQE), file=score)
                save_path = os.path.join("OutputImages", args.dataset)
                output_image = image_name.split('.')[0] + '_ULAP.jpg'
                cv2.imwrite(os.path.join(save_path, output_image), sceneRadiance)
                print("Done!")
    
    time_end = time.time()
    if args.score != 'None':
        print("TIME COST = {0}".format(time_end-time_start), file=score)
        score.close()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--score', default='None', help='score log')
    parser.add_argument('--dataset', default='RUCB/A2-05', help='Input image directory')
    args = parser.parse_args()
    run(args)



import os
import datetime
import numpy as np
import cv2
import natsort
import argparse
import time

from CloseDepth import closePoint
from F_stretching import StretchingFusion
from MapFusion import Scene_depth
from MapOne import max_R
from MapTwo import R_minus_GB
from blurrinessMap import blurrnessMap
from getAtomsphericLightFusion import ThreeAtomsphericLightFusion
from getAtomsphericLightOne import getAtomsphericLightDCP_Bright
from getAtomsphericLightThree import getAtomsphericLightLb
from getAtomsphericLightTwo import getAtomsphericLightLv
from getRGbDarkChannel import getRGB_Darkchannel
from getRefinedTransmission import Refinedtransmission
from getTransmissionGB import getGBTransmissionESt
from getTransmissionR import getTransmission
from global_Stretching import global_stretching
from sceneRadiance import sceneRadianceRGB
from sceneRadianceHE import RecoverHE
from evaluation import getScore

# 恢复原图像
# img : BGR格式
def getRecoverScene(img, blockSize=9, n=5, percent=0.001):
    RGB_Darkchannel = getRGB_Darkchannel(img, blockSize)
    BlurrnessMap = blurrnessMap(img, blockSize, n)
    AtomsphericLightOne = getAtomsphericLightDCP_Bright(RGB_Darkchannel, img, percent)
    AtomsphericLightTwo = getAtomsphericLightLv(img)
    AtomsphericLightThree = getAtomsphericLightLb(img, blockSize, n)
    AtomsphericLight = ThreeAtomsphericLightFusion(AtomsphericLightOne, AtomsphericLightTwo, AtomsphericLightThree, img)
    print('AtomsphericLight',AtomsphericLight)   # [b,g,r]


    R_map = max_R(img, blockSize)
    mip_map = R_minus_GB(img, blockSize, R_map)
    bluriness_map = BlurrnessMap

    d_R = 1 - StretchingFusion(R_map)
    d_D = 1 - StretchingFusion(mip_map)
    d_B = 1 - StretchingFusion(bluriness_map)

    d_n = Scene_depth(d_R, d_D, d_B, img, AtomsphericLight)
    d_n_stretching = global_stretching(d_n)
    d_0 = closePoint(img, AtomsphericLight)
    d_f = 8  * (d_n +  d_0)

    transmissionR = getTransmission(d_f)
    transmissionB, transmissionG = getGBTransmissionESt(transmissionR, AtomsphericLight)
    transmissionB, transmissionG, transmissionR = Refinedtransmission(transmissionB, transmissionG, transmissionR, img)

    sceneRadiance = sceneRadianceRGB(img, transmissionB, transmissionG, transmissionR, AtomsphericLight)

    return sceneRadiance

def run(args):
    root_path = "/home/tongjiayan/UnderwaterImageRestoration/densedepth_based_restoration/datasets"
    dataset_path = os.path.join(root_path, args.dataset)
    score = open(args.score, mode = 'a',encoding='utf-8')
    np.seterr(over='raise') # 触发FloatingPointError
    time_start = time.time()
    for image_name in os.listdir(dataset_path):
        image_path = os.path.join(dataset_path, image_name)
        if os.path.isfile(image_path):
            print('********    file   ********',image_name)
            img = cv2.imread(image_path) # BGR格式
            img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 转换成RGB格式
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
                sceneRadiance_RGB = cv2.cvtColor(sceneRadiance, cv2.COLOR_BGR2RGB)
                UICM, UISM, UIConM, UIQM, UCIQE = getScore(np.array(sceneRadiance_RGB))
                print("{0}\t{1}\t{2}\t{3}\t{4}".format(UICM, UISM, UIConM, UIQM, UCIQE), file=score)
                save_path = os.path.join("OutputImages", args.dataset)
                output_image = image_name.split('.')[0] + '_IBLA.jpg'
                cv2.imwrite(os.path.join(save_path, output_image), sceneRadiance)
                print("Done!")
    
    time_end = time.time()
    print("TIME COST = {0}".format(time_end-time_start), file=score)
    score.close()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--score', default='score.log', help='score log')
    parser.add_argument('--dataset', default='SQUID', help='Input image directory')
    args = parser.parse_args()
    run(args)
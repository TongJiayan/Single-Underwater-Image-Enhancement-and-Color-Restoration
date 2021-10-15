import os
import datetime
import numpy as np
import cv2
import natsort
import argparse
import time


from DetermineDepth import determineDepth
from TransmissionEstimation import getTransmission
from getAdaptiveExposureMap import AdaptiveExposureMap
from getAdaptiveSceneRadiance import AdaptiveSceneRadiance
from getAtomsphericLight import getAtomsphericLight
from refinedTransmission import refinedtransmission

from sceneRadianceGb import sceneRadianceGB
from sceneRadianceR import sceneradiance

from evaluation import getScore

# # # # # # # # # # # # # # # # # # # # # # Normalized implement is necessary part as the fore-processing   # # # # # # # # # # # # # # # #
# 恢复原图像
# img : BGR格式
def getRecoverScene(img, blockSize=9, n=5):
    img = (img - img.min()) / (img.max() - img.min()) * 255
    blockSize = 9
    largestDiff = determineDepth(img, blockSize)
    AtomsphericLight, AtomsphericLightGB, AtomsphericLightRGB = getAtomsphericLight(largestDiff, img)
    print('AtomsphericLightRGB',AtomsphericLightRGB)
    
    # transmission = getTransmission(img, AtomsphericLightRGB, blockSize=blockSize)
    transmission = getTransmission(img, AtomsphericLightRGB, blockSize)
    # print('transmission.shape',transmission.shape)
    # TransmissionComposition(folder, transmission, number, param='coarse')
    transmission = refinedtransmission(transmission, img)
    sceneRadiance_GB = sceneRadianceGB(img, transmission, AtomsphericLightRGB)

    # cv2.imwrite('OutputImages/' + prefix + 'GBDehazed.jpg', sceneRadiance_GB)


    # # print('sceneRadiance_GB',sceneRadiance_GB)
    sceneRadiance = sceneradiance(img, sceneRadiance_GB)
    S_x = AdaptiveExposureMap(img, sceneRadiance, Lambda=0.3, blockSize=blockSize)
    # print('S_x',S_x)
    sceneRadiance = AdaptiveSceneRadiance(sceneRadiance, S_x)

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
                output_image = image_name.split('.')[0] + '_GBdehazing.jpg'
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


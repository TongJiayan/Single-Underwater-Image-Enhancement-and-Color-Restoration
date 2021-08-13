
import os
import numpy as np
import cv2
import natsort
import xlwt
import argparse
import time

from skimage import exposure
from sceneRadianceCLAHE import RecoverCLAHE
from sceneRadianceHE import RecoverHE
from evaluation import getScore

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
                sceneRadiance = RecoverHE(img)
            except FloatingPointError:
                print("Fail to recover " + image_name + "!")
            else:
                sceneRadiance_RGB = cv2.cvtColor(sceneRadiance, cv2.COLOR_BGR2RGB)
                UICM, UISM, UIConM, UIQM, UCIQE = getScore(np.array(sceneRadiance_RGB))
                print("{0}\t{1}\t{2}\t{3}\t{4}".format(UICM, UISM, UIConM, UIQM, UCIQE), file=score)
                save_path = os.path.join("OutputImages", args.dataset)
                output_image = image_name.split('.')[0] + '_HE.jpg'
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
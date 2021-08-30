# Created by Joey
# log2xlsx

import os
import pandas as pd
import copy
import argparse

def log2xlsx(args):
    with open(args.score_log, 'r', encoding='UTF-8') as log:
        record = []
        before_score = []
        line = log.readline() # UICM-UISM-UIConM-UIQM-UCIQE
        line = log.readline()
        
        while line:
            if line[0:4]=="Done":
                break
            elif line[-4:]=="png\n": # image name's suffix
                record.clear()
                record.append(line.strip('\n'))
            else:
                #record = record + line.strip('\n').split('\t')
                record = record + line.strip('\n').split('\n')
                before_score.append(copy.deepcopy(record))  
            print(record) 
            line = log.readline()
        
    #columns = ["image_name", "PSNR", "SSIM"]
    columns = ["image_name", "MSE"]
    before_dt = pd.DataFrame(before_score, columns=columns)
    before_dt.to_excel(args.score_xlsx, index=0)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--score-log', default='MSE_HE.log', help='score log')
    parser.add_argument('--score-xlsx', default='MSE_HE.xlsx', help='score xlsx')
    
    args = parser.parse_args()
    log2xlsx(args)

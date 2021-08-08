import numpy as np


def getMinChannel(img):
	# 输入检查
    if len(img.shape) == 3 and img.shape[2] == 3:
        pass
    else:
        print("bad image shape, input must be color image")
        return None
    imgGray = img[:,:,1:3].min(axis=2) # R G B 中只考虑G B两个颜色通道 

    return imgGray

def getDarkChannel(img, blockSize):
    img = getMinChannel(img)

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
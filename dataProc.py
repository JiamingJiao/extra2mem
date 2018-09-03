#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2 as cv
import glob
import shutil
import math
import random

DATA_TYPE = np.float64

def npyToPng(srcDir, dstDir):
    src = loadData(srcDir = srcDir, normalization = True, normalizationRange = (0, 255))
    for i in range(0, src.shape[0]):
        cv.imwrite(dstDir + "%06d"%i+".png", src[i, :, :])
    print('convert .npy to .png completed')

def loadData(srcDir, resize=False, srcSize=(200, 200), dstSize=(256, 256), normalization=False, normalizationRange=(0, 1), approximateData=False):
    srcPathList = sorted(glob.glob(srcDir + '*.npy'))
    lowerBound = normalizationRange[0]
    upperBound = normalizationRange[1]
    if resize == False:
        dstSize = srcSize
    dst = np.ndarray((len(srcPathList), dstSize[0], dstSize[1]), dtype = DATA_TYPE)
    src = np.ndarray(srcSize, dtype = DATA_TYPE)
    index = 0
    for srcPath in srcPathList:
        src = np.load(srcPath)
        if resize == True:
            dst[index] = cv.resize(src, dstSize, dst[index], 0, 0, cv.INTER_NEAREST)
        else:
            dst[index] = src
        index += 1
    if approximateData == True:
        min = np.amin(dst)
        max = np.amax(dst)
        dst = 255*(dst-min)/(max-min)
        dst = np.around(dst)
        normalization = True
    if normalization == True:
        min = np.amin(dst)
        max = np.amax(dst)
        dst = lowerBound + ((dst-min)*(upperBound-lowerBound))/(max-min)
    return dst

def calcPseudoEcg(src, dst): # src: extra cellular potential map, dst: pseudo-ECG map
    dst = np.ndarray(src.shape, dtype = DATA_TYPE)
    diffVKernel = np.zeros((3, 3, 1), dtype = DATA_TYPE)
    diffVKernel[1, :, 0] = 1
    diffVKernel[:, 1, 0] = 1
    diffVKernel[1, 1, 0] = -4
    diffV = np.ndarray(src.shape, dtype = DATA_TYPE)
    diffV = cv.filter2D(src = src[i], ddepth = -1, kernel = diffVKernel, dst = diffV, anchor = (-1, -1), delta = 0, borderType = cv.BORDER_REPLICATE)
    distance = np.ndarray(src.shape, dtype = DATA_TYPE)
    dst = np.ndarray(src.shape, dtype = DATA_TYPE)
    firstRowIndex = np.linspace(0, src.shape[0], num = src.shape[1], endpoint = False)
    firstColIndex = np.linspace(0, src.shape[1], num = src.shape[1], endpoint = False)
    colIndex, rowIndex = np.meshgrid(firstRowIndex, firstColIndex)
    for row in range(0, src.shape[0]):
        for col in range(0, src.shape[1]):
            distance = cv.magnitude((rowIndex-row), (colIndex-col))
            dst[row,col] = cv.sumElems(cv.divide(diffV, distance))[0]
    return dst

def downSample(src, dst, samplePoints = (20, 20)):
    rowStride = math.floor(src.shape[0]/samplePoints[0])
    colStride = math.floor(src.shape[1]/samplePoints[1])
    multipleOfStride = ((samplePoints[0]-1)*rowStride+1, (samplePoints[1]-1)*colStride+1)


'''
def generatePseudoECG(srcDir, dstDir):
    src = loadData(srcDir = srcDir)
    dst = np.ndarray(src.shape, dtype = DATA_TYPE)
    diffVKernel = np.zeros((3, 3, 1), dtype = DATA_TYPE)
    diffVKernel[1, :, 0] = 1
    diffVKernel[:, 1, 0] = 1
    diffVKernel[1, 1, 0] = -4
    diffV = np.ndarray((src.shape[1], src.shape[2]), dtype = DATA_TYPE)
    distance = np.ndarray((src.shape[1], src.shape[2]), dtype = DATA_TYPE)
    pseudoECG = np.ndarray((src.shape[1], src.shape[2]), dtype = DATA_TYPE)
    firstRowIndex = np.linspace(0, src.shape[1], num = src.shape[1], endpoint = False)
    firstColIndex = np.linspace(0, src.shape[2], num = src.shape[2], endpoint = False)
    colIndex, rowIndex = np.meshgrid(firstRowIndex, firstColIndex)
    for i in range(0, src.shape[0]):
        diffV = cv.filter2D(src = src[i], ddepth = -1, kernel = diffVKernel, dst = diffV, anchor = (-1, -1), delta = 0, borderType = cv.BORDER_REPLICATE)
        for row in range(0, src.shape[1]):
            for col in range(0, src.shape[2]):
                distance = cv.magnitude((rowIndex-row), (colIndex-col))
                pseudoECG[row,col] = cv.sumElems(cv.divide(diffV, distance))[0]
        dstFileName = dstPath + '%06d'%i
        np.save(dstFileName, pseudoECG)
    print('completed')
'''

def downSample(srcPath, dstPath, samplePoints = (5, 5), interpolationSize = (200, 200)):
    src = loadData(srcPath, approximateData = False)
    rowStride = math.floor(src.shape[1]/samplePoints[0])
    colStride = math.floor(src.shape[2]/samplePoints[1])
    multipleOfStride = ((samplePoints[0]-1)*rowStride+1, (samplePoints[1]-1)*colStride+1)
    temp = np.ndarray(multipleOfStride, dtype = DATA_TYPE) #Its size is a multiple of stride + 1
    sample = np.ndarray(samplePoints, dtype = DATA_TYPE)
    interpolated = np.ndarray(interpolationSize, dtype = DATA_TYPE)
    for i in range(0, src.shape[0]):
        temp = cv.resize(src[i, :, :], multipleOfStride)
        for j in range(0, samplePoints[0]):
            for k in range(0, samplePoints[1]):
                sample[j, k] = temp[j*rowStride, k*colStride]
        interpolated = cv.resize(sample, interpolationSize, interpolated, 0, 0, cv.INTER_NEAREST)
        dstFileName = dstPath + '%06d'%i
        np.save(dstFileName, interpolated)
    print('down sampling completed')

def generateSparsePseudoECG(srcPath, dstPath, samplePoints = (10, 10)):
    src = loadData(srcPath)
    rowStride = math.floor(src.shape[1]/samplePoints[0])
    colStride = math.floor(src.shape[2]/samplePoints[1])
    multipleOfStride = ((samplePoints[0]-1)*rowStride+1, (samplePoints[1]-1)*colStride+1)
    temp = np.ndarray(multipleOfStride, dtype = DATA_TYPE) #Its size is a multiple of stride
    sample = np.ndarray(samplePoints, dtype = DATA_TYPE)
    diffVKernel = np.zeros((3, 3, 1), dtype = DATA_TYPE)
    diffVKernel[1, :, 0] = 1
    diffVKernel[:, 1, 0] = 1
    diffVKernel[1, 1, 0] = -4
    diffV = np.ndarray(multipleOfStride, dtype = DATA_TYPE)
    firstRowIndex = np.linspace(0, temp.shape[0], num = temp.shape[1], endpoint = False)
    firstColIndex = np.linspace(0, temp.shape[0], num = temp.shape[1], endpoint = False)
    colIndex, rowIndex = np.meshgrid(firstRowIndex, firstColIndex)
    distance = np.ndarray(multipleOfStride, dtype = DATA_TYPE)
    pseudoECG = np.ndarray(samplePoints, dtype = DATA_TYPE)
    interpolated = np.ndarray((src.shape[1], src.shape[2]), dtype = DATA_TYPE)
    for i in range(0, src.shape[0]):
        #diffV = cv.filter2D(src = src[i], ddepth = -1, kernel = diffVKernel, dst = diffV, anchor = (-1, -1), delta = 0, borderType = cv.BORDER_REPLICATE)
        temp = cv.resize(src[i, :, :], multipleOfStride, temp, 0, 0, cv.INTER_CUBIC)
        diffV = cv.filter2D(src = temp, ddepth = -1, kernel = diffVKernel, dst = diffV, anchor = (-1, -1), delta = 0, borderType = cv.BORDER_REPLICATE)
        for row in range(0, samplePoints[0]):
            for col in range(0, samplePoints[1]):
                distance = cv.magnitude((rowIndex-row*rowStride), (colIndex-col*colStride))
                pseudoECG[row, col] = cv.sumElems(cv.divide(diffV, distance))[0]
        interpolated = cv.resize(pseudoECG, (src.shape[1], src.shape[2]), interpolated, 0, 0, cv.INTER_NEAREST)
        np.save(dstPath + '%06d'%i, interpolated)

def loadImage(srcPath, resize = 0, rawRows = 200, rawCols = 200, imgRows = 256, imgCols = 256, normalization = 0):
    fileName = glob.glob(srcPath + '*.png')
    if resize == 0:
        mergeImg = np.ndarray((len(fileName), rawRows, rawCols), dtype = DATA_TYPE)
    else:
        mergeImg = np.ndarray((len(fileName), imgRows, imgCols), dtype = DATA_TYPE)
        tempImg = np.ndarray((imgRows, imgCols), dtype = DATA_TYPE)
    rawImg = np.ndarray((rawRows, rawCols), dtype = DATA_TYPE)
    for i in range(0, len(fileName)):
        localName = srcPath + '%06d'%i + ".png"
        rawImg = cv.imread(localName, -1)
        if resize == 1:
            mergeImg[i] = cv.resize(rawImg, (imgRows, imgCols))
        else:
            mergeImg[i] = rawImg
    if normalization == 1:
        min = np.amin(mergeImg)
        max = np.amax(mergeImg)
        mergeImg = (mergeImg-min)/(max-min)
    return mergeImg

def create3DData(src, temporalDepth):
    framesNum = src.shape[0]
    paddingDepth = math.floor((temporalDepth-1)/2 + 0.1)
    dst = np.zeros((framesNum, temporalDepth, src.shape[1], src.shape[2]), dtype = DATA_TYPE)
    for i in range(0, paddingDepth):
        dst[i, paddingDepth-i:temporalDepth, :, :] = src[0:temporalDepth-paddingDepth+i, :, :]
        dst[framesNum-1-i, 0:temporalDepth-paddingDepth+i, :, :] = src[framesNum-(temporalDepth-paddingDepth)-i:framesNum, :, :]
    for i in range(paddingDepth, framesNum-paddingDepth):
        dst[i, :, :, :] = src[i-paddingDepth:i+paddingDepth+1, :, :]
    return dst

def clipData(srcPath, dstPath, bounds = [0., 1.]):
    src = loadData(srcPath)
    dst = np.clip(src, bounds[0], bounds[1])
    for i in range(0, src.shape[0]):
        dstFileName = dstPath + '%06d'%i
        np.save(dstFileName, dst[i])

def splitTrainAndVal(src1, src2, valSplit):
    srcLength = src1.shape[0]
    dataType = src1.dtype
    dimension1 = src1.ndim
    dimension2 = src2.ndim
    valNum = math.floor(valSplit*srcLength+0.1)
    randomIndexes = random.sample(range(0, srcLength), valNum)
    trainDataShape1 = np.ndarray((dimension1), dtype = np.uint8)
    valDataShape1 = np.ndarray((dimension1), dtype = np.uint8)
    trainDataShape1[0] = srcLength - valNum
    valDataShape1[0] = valNum
    trainDataShape1[1:dimension1] = src1.shape[1:dimension1]
    valDataShape1[1:dimension1] = src1.shape[1:dimension1]
    trainDataShape2 = np.ndarray((dimension2), dtype = np.uint8)
    valDataShape2 = np.ndarray((dimension2), dtype = np.uint8)
    trainDataShape2[0] = srcLength - valNum
    valDataShape2[0] = valNum
    trainDataShape2[1:dimension2] = src2.shape[1:dimension2]
    valDataShape2[1:dimension2] = src2.shape[1:dimension2]
    dst = [np.ndarray((trainDataShape1), dtype = dataType), np.ndarray((valDataShape1), dtype = dataType),
    np.ndarray((trainDataShape2), dtype = dataType), np.ndarray((valDataShape2), dtype = dataType)]
    dst[1] = np.take(src1, randomIndexes, 0)
    dst[0] = np.delete(src1, randomIndexes, 0)
    dst[3] = np.take(src2, randomIndexes, 0)
    dst[2] = np.delete(src2, randomIndexes, 0)
    return dst

def copyMassiveData(srcPathList, dstPath, potentialName):
    startNum = 0
    for srcPath in srcPathList:
        fileName = sorted(glob.glob(srcPath + potentialName + '*.npy'))
        for srcName in fileName:
            dst = np.load(srcName)
            dstFileName = dstPath + '%06d'%startNum
            np.save(dstFileName, dst)
            startNum += 1

def copyData(srcPath, dstPath, startNum = 0, endNum = None, shiftNum = 0):
    fileName = sorted(glob.glob(srcPath + '*.npy'))
    del fileName[endNum+1:len(fileName)]
    del fileName[0:startNum]
    for srcName in fileName:
        dst = np.load(srcName)
        dstFileName = dstPath + '%06d'%(startNum+shiftNum)
        np.save(dstFileName, dst)
        startNum += 1

'''
def annealingDownSample(srcPath, dst, maskPath)
'''

# Do NOT import keras, tf or any other modules that require GPU in this file.

import numpy as np
import cv2 as cv
import glob
import os
import math
import random
import scipy.interpolate

DATA_TYPE = np.float32
CV_DATA_TYPE = cv.CV_32F
INTERPOLATION_METHOD = cv.INTER_NEAREST
NORM_RANGE = (0, 1)
VIDEO_ENCODER = 'XVID'
VIDEO_FPS = 20
IMG_SIZE = (200, 200)
PSEUDO_ECG_CONV_KERNEL = np.zeros((3, 3, 1), dtype=DATA_TYPE)
PSEUDO_ECG_CONV_KERNEL[1, :, 0] = 1
PSEUDO_ECG_CONV_KERNEL[:, 1, 0] = 1
PSEUDO_ECG_CONV_KERNEL[1, 1, 0] = -4
ECG_FOLDER_NAME = 'pseudo_ecg'


class SparsePecg(object):
    def __init__(self, shape, coordinates, roi=-1, d=0):
        self.coordinates = coordinates
        self.shape = shape
        firstRowIndex = np.linspace(0, shape[0], num=shape[0], endpoint=False, dtype=DATA_TYPE)
        firstColIndex = np.linspace(0, shape[1], num=shape[1], endpoint=False, dtype=DATA_TYPE)
        colIndex, rowIndex = np.meshgrid(firstRowIndex, firstColIndex)
        self.grid = (rowIndex, colIndex)
        self.distance = np.ndarray(((coordinates.shape[0],) + shape), dtype=DATA_TYPE)
        dArray = np.zeros(shape, dtype=DATA_TYPE)
        dArray[:, :] = d
        for i in range(0, coordinates.shape[0]):
            cv.magnitude((rowIndex - coordinates[i, 0]), (colIndex - coordinates[i, 1]), self.distance[i])
            cv.magnitude(self.distance[i], dArray, self.distance[i])
        self.pseudoEcg = np.ndarray(coordinates.shape[0], dtype=DATA_TYPE)
        self.diffV = np.ndarray(shape, dtype=DATA_TYPE)
        self.quotient = np.ndarray(shape, dtype=DATA_TYPE)
        self.dst = np.ndarray(shape, dtype=DATA_TYPE)
        if roi == -1:
            self.roi = [[0, 0], [shape[0]-1, shape[1]-1]]
        else:
            self.roi = roi
        self.dstInRoi = np.ndarray((self.roi[1][0]-self.roi[0][0], self.roi[1][1]-self.roi[0][1]), dtype=DATA_TYPE)

    def calcPecg(self, src):
        cv.filter2D(src=src, ddepth=CV_DATA_TYPE, kernel=PSEUDO_ECG_CONV_KERNEL, dst=self.diffV, anchor=(-1, -1), delta=0, borderType=cv.BORDER_REPLICATE)
        for i in range(0, self.coordinates.shape[0]):
            cv.divide(self.diffV, self.distance[i], dst=self.quotient)
            self.pseudoEcg[i] = cv.sumElems(self.quotient)[0]
        self.dst = scipy.interpolate.griddata(self.coordinates, self.pseudoEcg, self.grid, method='nearest')
        self.dstInRoi = np.copy(self.dst[self.roi[0][0]:self.roi[1][0], self.roi[0][1]:self.roi[1][1]])

    def callCalc(self, srcDirList, dstDirList):
        length = len(srcDirList)
        for i in range(0, length):
            srcPath = srcDirList[i]
            src = loadData(srcPath)
            dstPath = dstDirList[i]
            if not os.path.exists(dstPath):
                os.mkdir(dstPath)
            for i in range(0, src.shape[0]):
                self.calcPecg(src[i])
                np.save(dstPath+'%06d'%i, self.dstInRoi)
            print('%s completed'%dstDirList[i])


class AccurateSparsePecg(SparsePecg):
    def __init__(self, srcShape, removeNum, fullCoordinatesShape, parentCoordinates, coordinatesDir, **kwargs):
        self.srcShape = srcShape
        rowStride = math.floor(srcShape[0]/fullCoordinatesShape[0])
        colStride = math.floor(srcShape[1]/fullCoordinatesShape[1])
        self.multipleOfStride = ((fullCoordinatesShape[0]-1)*rowStride+1, (fullCoordinatesShape[1]-1)*colStride+1)
        coordinates = removePoints(parentCoordinates, removeNum)
        super(AccurateSparsePecg, self).__init__(self.multipleOfStride, coordinates, **kwargs)
        if not coordinatesDir == 'None':
            np.save(coordinatesDir, self.coordinates)
        else:
            print('coordinates were not saved!')
    
    def resizeAndCalc(self, srcDirList, dstDirList):
        resizedSrc = np.ndarray((self.multipleOfStride+(1,)), dtype=DATA_TYPE)
        resizedDst = np.ndarray((self.srcShape+(1,)), dtype=DATA_TYPE)
        length = len(srcDirList)
        for i in range(0, length):
            srcPath = srcDirList[i]
            src = loadData(srcPath)
            dstPath = dstDirList[i]
            if not os.path.exists(dstPath):
                os.mkdir(dstPath)
            for i in range(0, src.shape[0]):
                cv.resize(src[i], self.multipleOfStride, resizedSrc, 0, 0, cv.INTER_CUBIC)
                self.calcPecg(resizedSrc)
                cv.resize(self.dst, self.srcShape, resizedDst, 0, 0, INTERPOLATION_METHOD)
                np.save(dstPath+'%06d'%i, resizedDst)
            print('%s completed'%dstPath)


class Data(object):
    def __init__(self, groups, length, height, width, channels):
        self.twoD = np.zeros((groups, length, height, width, channels), dtype=DATA_TYPE)
        self.groups = groups
        self.length = length
        self.height = height
        self.width = width
        self.channels = channels
        self.range = [np.amin(self.twoD), np.amax(self.twoD)]
        self.train = None
        self.val = None
        self.rnn = None
    
    def setData(self, dirList):
        # load and normalize
        for i in range(0, len(dirList)):
            self.twoD[i, :, :, :, :] = loadData(dirList[i])

    def setData2(self, pathList):
        for i, path in enumerate(pathList):
            self.twoD[i, :, :, :, :] = np.load(path)

    def saveData(self, pathPrefix, vname):
        # vname: phie, vmem, pecg ...
        for i in range(0, self.groups):
            path = os.path.join(pathPrefix, '%06d'%i, vname)
            if not os.path.exists(path):
                os.makedirs(path)
            for j in range(0, self.length):
                np.save(os.path.join(path, '%06d'%j), self.twoD[i, j, :, :, :])

    def saveData2(self, pathPrefix, vname):
        path = os.path.join(pathPrefix, vname)
        if not os.path.exists(path):
            os.makedirs(path)
        for i, data in enumerate(self.twoD):
            np.save(os.path.join(path, '%06d'%i), data)
                
    def normalize(self, normalizationRange=NORM_RANGE):
        self.twoD, minValue, maxValue = normalize(self.twoD, normalizationRange=normalizationRange)
        self.range = [minValue, maxValue]

    def splitTrainAndVal(self, valSplit, srcType):
        trainingLength = math.floor(self.groups*valSplit)
        if srcType == 'rnn':
            self.train = self.rnn[0:trainingLength]
            self.val = self.rnn[trainingLength:]
        else:
            self.train = self.twoD[0:trainingLength]
            self.val = self.twoD[trainingLength:]

    def rotate(self):
        pass

    def saveImage(self, pathPrefix):
        data, _, _ = normalize(self.twoD, normalizationRange=(0, 255))
        data = data.astype(np.uint8)
        for i, group in enumerate(data):
            path = os.path.join(pathPrefix, '%06d'%i)
            if not os.path.exists(path):
                os.makedirs(path)
            for j, img in enumerate(group):
                cv.imwrite(os.path.join(path, '%06d.png'%j), img)


class Phie(Data):
    def __init__(self, coordinates, *args, **kwargs):
        super(Phie, self).__init__(*args, **kwargs)
        self.train = None
        self.val = None
        self.coordinates = coordinates
        self.ground = None
        self.rnn = None

    # def setGround(self, index): # set reference point (ground, phie=0)
    #     self.ground = index
    #     #np.subtract(self.twoD, self.twoD[:, :, self.coordinates[self.ground, 0], self.coordinates[self.ground, 1], :], out=self.twoD)
    #     # Why above line doesn't work?
    #     for i, sequence in enumerate(self.twoD):
    #         for j, frame in enumerate(sequence):
    #             np.subtract(frame, frame[self.coordinates[self.ground, 0], self.coordinates[self.ground, 1], :], out=self.twoD[i, j, :, :, :])

    def downSample(self):
        sampled = np.squeeze(self.twoD[:, :, self.coordinates[:, 0], self.coordinates[:, 1], :], 3)

        firstRowIndex = np.linspace(0, self.height, num=self.height, endpoint=False, dtype=DATA_TYPE)
        firstColIndex = np.linspace(0, self.width, num=self.width, endpoint=False, dtype=DATA_TYPE)
        colIndex, rowIndex = np.meshgrid(firstRowIndex, firstColIndex)
        grid = (rowIndex, colIndex)

        for i, sequence in enumerate(self.twoD):
            for j, frame in enumerate(sequence):
                self.twoD[i, j, :, :, 0]=scipy.interpolate.griddata(self.coordinates, sampled[i, j], grid, method='nearest')

    def setRnnData(self, temporalDepth):
        validLength = self.length - temporalDepth + 1
        shape = ((self.groups, validLength, temporalDepth) + self.twoD.shape[2:])
        strides = (self.twoD.strides[0:2] + (self.twoD.strides[1],) + self.twoD.strides[2:])  # Expand dim_of_length to dim_of_length * dim_of_temporalDepth
        self.rnn = np.lib.stride_tricks.as_strided(self.twoD, shape=shape, strides=strides, writeable=False)  # (groups, validLength, temporalDepth, height, width, channels)


class Pecg(Phie):
    def __init__(self, *args, **kwargs):
        super(Pecg, self).__init__(*args, **kwargs)


class Vmem(Data):
    def __init__(self, *args, **kwargs):
        super(Vmem, self).__init__(*args, **kwargs)
        self.rnn = None

    def setRnnData(self, temporalDepth):
        self.rnn = self.twoD[:, temporalDepth-1:, :, :, :]  # (groups, validLength, height, width, channels)


def selectData(length, size, nameList, srcPathPrefix, dstPathPrefix):
    i = 0
    dstPath = os.path.join(dstPathPrefix, '%d_%d_%d'%(size[0], size[1], length))
    for name in nameList:
        srcPhie = loadData(os.path.join(srcPathPrefix, name, 'phie_'))
        srcVmem = loadData(os.path.join(srcPathPrefix, name, 'vmem_'))
        for j in range(length, srcPhie.shape[0], length):
            for x in range(size[0], srcPhie.shape[1], size[0]):
                for y in range(size[1], srcPhie.shape[2], size[1]):
                    dstPhiePath = os.path.join(dstPath, ''.join(['%06d_'%i, name]), 'phie')
                    dstVmemPath = os.path.join(dstPath, ''.join(['%06d_'%i, name]), 'vmem')
                    if not os.path.exists(dstPhiePath):
                        os.makedirs(dstPhiePath)
                    if not os.path.exists(dstVmemPath):
                        os.makedirs(dstVmemPath)
                    x0 = x-size[0]
                    y0 = y-size[1]
                    for k in range(1, length+1):
                        np.save(os.path.join(dstPhiePath, '%06d.npy'%(length-k)), srcPhie[j-k, x0:x, y0:y])
                        np.save(os.path.join(dstVmemPath, '%06d.npy'%(length-k)), srcVmem[j-k, x0:x, y0:y])
                    i += 1


def selectData2(size, nameList, srcPathPrefix, dstPathPrefix):
    i = 0
    dstPhiePath = os.path.join(dstPathPrefix, '%d_%d'%size, 'phie')
    dstVmemPath = os.path.join(dstPathPrefix, '%d_%d'%size, 'vmem')
    if not os.path.exists(dstPhiePath):
        os.makedirs(dstPhiePath)
    if not os.path.exists(dstVmemPath):
        os.makedirs(dstVmemPath)
    for name in nameList:
        srcPhie = loadData(os.path.join(srcPathPrefix, name, 'phie_'))
        srcVmem = loadData(os.path.join(srcPathPrefix, name, 'vmem_'))
        for x in range(size[0], srcPhie.shape[1], size[0]):
            for y in range(size[1], srcPhie.shape[2], size[1]):
                np.save(os.path.join(dstPhiePath, ''.join(['%06d_'%i, name])), srcPhie[:, x-size[0]:x, y-size[1]:y])
                np.save(os.path.join(dstVmemPath, ''.join(['%06d_'%i, name])), srcVmem[:, x-size[0]:x, y-size[1]:y])
                i += 1


def selectData3(length, srcPathPrefix, dstPathPrefix):
    dstPhiePath = os.path.join(dstPathPrefix, 'phie')
    dstVmemPath = os.path.join(dstPathPrefix, 'vmem')
    if not os.path.exists(dstPhiePath):
        os.makedirs(dstPhiePath)
    if not os.path.exists(dstVmemPath):
        os.makedirs(dstVmemPath)
    srcPhiePathList = sorted(glob.glob(os.path.join(srcPathPrefix, 'phie', '*.npy')))
    srcVmemPathList = sorted(glob.glob(os.path.join(srcPathPrefix, 'vmem', '*.npy')))
    i = 0
    for j in range(0, len(srcPhiePathList)):
        srcPhie = np.load(srcPhiePathList[j])
        srcVmem = np.load(srcVmemPathList[j])
        for k in range(length, srcPhie.shape[0], length):
            np.save(os.path.join(dstPhiePath, '%06d'%i), srcPhie[k-length:k])
            np.save(os.path.join(dstVmemPath, '%06d'%i), srcVmem[k-length:k])
            i += 1


def shuffleData(ecg, vmem):
    # shuffle on the dim of groups
    state = np.random.get_state()
    np.random.shuffle(ecg)
    np.random.set_state(state)
    np.random.shuffle(vmem)


def createDirList(dataDir, nameList, potentialName=''):
    length = len(nameList)
    dst = [None]*length
    for i in range(0, length):
        dst[i] = dataDir + nameList[i] + '/' + potentialName
    return dst


def randomCoordinates(pointsNum, limit):
    sampledPoints = random.sample(range(0, limit[0]*limit[1]), pointsNum)
    dst = np.ndarray((pointsNum, 2), dtype=np.uint16)
    dst[:, 0], dst[:, 1] = np.divmod(sampledPoints, limit[1])
    return dst


def uniformCoordinates(shape, limit):
    firstRowIndex = np.linspace(0, limit[0]-1, num=shape[0], dtype=DATA_TYPE)
    firstColIndex = np.linspace(0, limit[1]-1, num=shape[1], dtype=DATA_TYPE)
    colIndex, rowIndex = np.meshgrid(firstRowIndex, firstColIndex)
    dst = np.ndarray((shape[0]*shape[1], 2), dtype=np.uint16)
    dst[:, 0] = rowIndex.flatten()
    dst[:, 1] = colIndex.flatten()
    return dst


def removePoints(src, num=0):
    deletedPoints = random.sample(range(0, src.shape[0]), num)
    dst = np.delete(src, deletedPoints, 0)
    return dst


def clipData(src, bounds=(0, 1)):
    dst = np.clip(src, bounds[0], bounds[1])
    return dst


def npyToPng(srcDir, dstDir):
    if not os.path.exists(dstDir):
        os.mkdir(dstDir)
    src = loadData(srcDir=srcDir)
    src, _, _ = normalize(src, (0, 255))
    for i in range(0, src.shape[0]):
        cv.imwrite(dstDir + "%06d"%i+".png", src[i, :, :])
    print('convert .npy to .png completed')


def loadData(srcDir, resize=False, dstSize=(256, 256), withChannel=True):  # doesn't work with multi-channel image
    srcPathList = sorted(glob.glob(srcDir + '*.npy'))
    if len(srcPathList)==0:
        srcPathList = sorted(glob.glob(srcDir + '/*.npy'))
    length = len(srcPathList)
    sample = np.load(srcPathList[0])
    src = np.ndarray(sample.shape, dtype=DATA_TYPE)
    if not resize:
        dstSize = (sample.shape[0], sample.shape[1])
    if sample.ndim == 2:
        dst = np.ndarray((length,)+dstSize, dtype=DATA_TYPE)
    elif sample.ndim == 3:
        if resize:
            dst = np.ndarray(((length,)+dstSize), dtype=DATA_TYPE)
        else:
            dst = np.ndarray(((length,)+sample.shape), dtype=DATA_TYPE)
    for i in range(0, length):
        src = np.load(srcPathList[i])
        src = src.astype(DATA_TYPE)
        if resize:
            cv.resize(src, dstSize, dst[i], 0, 0, INTERPOLATION_METHOD)
        else:
            dst[i] = src
    if withChannel:
        dst = dst.reshape((length,)+dstSize+(1,))
    else:
        dst = dst.reshape((length,)+dstSize)
    return dst


def create3dSequence(srcEcg, srcMem, temporalDepth, netGName):
    dstEcg = create3dEcg(srcEcg, temporalDepth, netGName)
    if netGName == 'convLstm':
        startFrame = temporalDepth
    if netGName == 'uNet3d':
        startFrame = math.floor(temporalDepth/2 + 0.1)
    dstMem = srcMem[startFrame:dstEcg.shape[0]+startFrame]
    return dstEcg, dstMem


def create3dEcg(src, temporalDepth, netGName):
    if netGName == 'convLstm':
        framesNum = src.shape[0]-temporalDepth
    elif netGName == 'uNet3d':
        framesNum = src.shape[0] - 2*math.floor(temporalDepth/2 + 0.1)
    dst = np.zeros((framesNum, temporalDepth, src.shape[1], src.shape[2], src.shape[3]), dtype=DATA_TYPE)
    for i in range(0, framesNum):
        dst[i] = src[i:i+temporalDepth]
    return dst


def mergeSequence(pecgDirList, memDirList, temporalDepth, netGName=None, srcSize=(200, 200), dstSize=(256, 256), normalizationRange=NORM_RANGE):
    length = len(pecgDirList)
    ecg = np.empty(length, dtype=object)
    mem = np.empty(length, dtype=object)
    for i in range(0, length):
        srcEcg = loadData(srcDir=pecgDirList[i], resize=True, dstSize=dstSize)
        srcMem = loadData(srcDir=memDirList[i], resize=True, dstSize=dstSize)
        if netGName=='convLstm' or netGName=='uNet3d':
            ecg[i], mem[i] = create3dSequence(srcEcg, srcMem, temporalDepth, netGName)
        if netGName=='uNet':
            ecg[i] = srcEcg
            mem[i] = srcMem
    del srcEcg, srcMem
    ecg = np.concatenate(ecg)
    mem = np.concatenate(mem)
    ecg, minEcg, maxEcg = normalize(ecg, normalizationRange)
    mem, minMem, maxMem = normalize(mem, normalizationRange)
    dataRange = [minEcg, maxEcg, minMem, maxMem]
    print('min ecg is %.8f'%minEcg)
    print('max ecg is %.8f'%maxEcg)
    print('min mem is %.8f'%minMem)
    print('max mem is %.8f'%maxMem)
    return [ecg, mem, dataRange]


def copyMassiveData(srcDirList, dstDir, potentialName):
    startNum = 0
    for srcDir in srcDirList:
        fileName = sorted(glob.glob(srcDir + potentialName + '*.npy'))
        for srcName in fileName:
            dst = np.load(srcName)
            np.save(dstDir + '%06d'%startNum, dst)
            startNum += 1


def copyData(srcPath, dstPath, startNum=0, endNum=None, shiftNum=0):
    fileName = sorted(glob.glob(srcPath + '*.npy'))
    del fileName[endNum+1:len(fileName)]
    del fileName[0:startNum]
    for srcName in fileName:
        dst = np.load(srcName)
        dstFileName = dstPath + '%06d'%(startNum+shiftNum)
        np.save(dstFileName, dst)
        startNum += 1


def normalize(src, normalizationRange=NORM_RANGE):
    minValue = src.min()
    maxValue = src.max()
    dst = np.empty_like(src)
    np.add(src, -minValue, dst)
    factor = (normalizationRange[1]-normalizationRange[0]) / (maxValue-minValue)
    np.multiply(dst, factor, dst)
    return [dst, minValue, maxValue]


def channelNormalize(src):
    min_arr = np.amin(src, 0)
    max_arr = np.amax(src, 0)
    range_arr_ = max_arr - min_arr
    eps = np.finfo(np.float32).eps
    range_arr = np.where(range_arr_<eps, 1, range_arr_)
    dst = (src - min_arr) / range_arr
    return dst


def scale(src, priorRange=None, dstRange=(0, 1)):
    dst = dstRange[0] + ((src-priorRange[0])*(dstRange[1]-dstRange[0])) / (priorRange[1]-priorRange[0])
    return dst


def makeVideo(srcDir, dstPath, frameRange=(-1, -1), padding=(0, 0), fps=VIDEO_FPS, frameSize=IMG_SIZE):
    # color
    srcPathList = sorted(glob.glob(srcDir+'*.png'))
    if not frameRange[0] == -1:
        srcPathList = srcPathList[frameRange[0]:]
    if not frameRange[1] == -1:
        srcPathList = srcPathList[:frameRange[1]-frameRange[0]]
    sample = cv.imread(srcPathList[0], -1)
    paddingArray = np.zeros_like(sample)
    writer = cv.VideoWriter(filename=dstPath, fourcc=cv.VideoWriter_fourcc(*VIDEO_ENCODER), fps=fps, frameSize=frameSize, isColor=True)
    for i in range(0, padding[0]):
        writer.write(paddingArray)
    for i in srcPathList:
        src = cv.imread(i, -1)
        writer.write(src)
    for i in range(0, padding[1]):
        writer.write(paddingArray)
    writer.release


def makeVideo2(src, dstPath, padding=(0, 0), fps=VIDEO_FPS, frameSize=IMG_SIZE):
    src = np.copy(src)
    if src.dtype == np.uint8:
        bgr = src
    else:
        bgr = np.empty(src.shape[0:3]+(3,), np.float32)
        if src.ndim == 3:
            # no channel dim
            bgr[:, :, :, :] = src[:, :, :, np.newaxis]
        elif src.ndim == 4:
            if src.shape[3] == 1:
                bgr[:, :, :, :] = src
            elif src.shape[3]==3:
                for i, frame in enumerate(src):
                    cv.cvtColor(frame, cv.COLOR_RGB2BGR, bgr[i])
            elif src.shape[3]==4:
                for i, frame in enumerate(src):
                    cv.cvtColor(frame, cv.COLOR_RGBA2BGR, bgr[i])
        bgr *= 255
        bgr = bgr.astype(np.uint8)
    writer = cv.VideoWriter(filename=dstPath, fourcc=cv.VideoWriter_fourcc(*VIDEO_ENCODER), fps=fps, frameSize=frameSize, isColor=True)
    resized = np.zeros((frameSize + (3,)), np.uint8)
    paddingArray = np.zeros_like(resized)
    for i in range(0, padding[0]):
        writer.write(paddingArray)
    for frame in bgr:
        cv.resize(frame, frameSize, resized, 0, 0, cv.INTER_CUBIC)
        writer.write(resized)
    for i in range(0, padding[1]):
        writer.write(paddingArray)
    writer.release


def sequenceToBlocks(src, block_length, telomere_length=0):
    blocks_list = []
    for k in range(0, src.shape[0]-block_length+1, block_length-telomere_length*2):
        blocks_list.append(src[np.newaxis, k:k+block_length])
    dst = np.concatenate(blocks_list)
    return dst


def blocksToSequence(src, telomere_length=0):
    valid_block_length = src.shape[1] - telomere_length*2
    valid = src[:, telomere_length:telomere_length+valid_block_length, ...]
    dst = valid.reshape((src.shape[0]*valid_block_length,)+src.shape[2:5])
    return dst


def resizeSequence(src, dst_size, inter_method=INTERPOLATION_METHOD, dst=None):
    if dst is None:
        dst = np.zeros((src.shape[0], dst_size[0], dst_size[1], src.shape[-1]), src.dtype)
    for src_frame, dst_frame in zip(src, dst):
        cv.resize(src_frame, dst_size, dst_frame, interpolation=inter_method)
    return dst

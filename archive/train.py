import numpy as np
import os
import datetime
import pickle
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import dataProc
import utils
from model import Networks
from analysis import calculateMae


class Generator(Networks):
    def __init__(self, netgName, batchSize=10, **networksArgs):
        super(Generator, self).__init__(**networksArgs)
        getModel = {'uNet': self.uNet,
                    'uNet5': self.uNet5,
                    'convLstm': self.convLstm,
                    'uNet3d': self.uNet3d,
                    'convLstm5': self.convLstm5,
                    'seqConv5': self.seqConv5,
                    'uNet3d5': self.uNet3d5}
        self.netg = getModel[netgName]()
        self.netg.summary()
        self.batchSize = batchSize
        self.dataRange = [0.0]*4  # minPecg, maxPecg, minVmem, maxVmem
        self.checkpointer = None
        self.earlyStopping = None
        self.learningRate = None
        self.pecg = None
        self.vmem = None
        self.history = None
        self.epochsNum = None
        self.valSplit = None
        self.callbacks = None

        self.prediction = None
        self.truth = None

    def train(self, pecg, vmem, learningRateG=1e-4, lossFuncG='mae', continueTrain=False, pretrainedModelPath=None,
              modelDir=None, earlyStoppingPatience=10, epochsNum=200, valSplit=0.2, constantLearningRate=False):
        self.pecg = pecg
        self.vmem = vmem
        self.epochsNum = epochsNum
        self.valSplit = valSplit
        self.netg.compile(optimizer=Adam(lr=learningRateG), loss=lossFuncG, metrics=[lossFuncG])
        if continueTrain:
            self.netg.load_weights(pretrainedModelPath)
        if not os.path.exists(modelDir):
            os.makedirs(modelDir)
        self.checkpointer = ModelCheckpoint(os.path.join(modelDir, 'netg.h5'), monitor='val_mean_absolute_error', verbose=1, save_best_only=True, save_weights_only=True,
                                            mode='min')
        self.learningRate = ReduceLROnPlateau('val_mean_absolute_error', 0.1, earlyStoppingPatience, 1, 'auto', 1e-4, min_lr=learningRateG*1e-4)
        if constantLearningRate:
            self.earlyStopping = EarlyStopping(patience=earlyStoppingPatience, verbose=1)
            self.callbacks = [self.checkpointer, self.earlyStopping]
        else:
            self.earlyStopping = EarlyStopping(patience=earlyStoppingPatience*5, verbose=1)
            self.callbacks = [self.checkpointer, self.earlyStopping, self.learningRate]
        # self.pecg = dataProc.Phie(None, len(pecgDirList), length, self.rawSize[0], self.rawSize[1], self.channels)
        # self.vmem = dataProc.Vmem(len(vmemDirList), length, self.rawSize[0], self.rawSize[1], self.channels)
        # self.pecg.setData2(pecgDirList)
        # self.vmem.setData2(vmemDirList)
        self.pecg.normalize()
        self.vmem.normalize()
        self.dataRange = self.pecg.range + self.vmem.range
        print(self.dataRange)
        np.save(os.path.join(modelDir, 'data_range'), np.array(self.dataRange))
        dataProc.shuffleData(self.pecg.twoD, self.vmem.twoD)
        trainingFunc = {
            'uNet': self.trainUNet,
            'uNet5': self.trainUNet,
            'convLstm': self.trainConvLstm,
            'convLstm5': self.trainConvLstm,
            'seqConv5': self.trainSeqConv5,
            'uNet3d': self.trainUNet3d,
            'uNet3d5': self.trainUNet3d}
        trainingFunc[self.netg.name]()
        with open(modelDir+'history', 'wb') as historyFile:
            pickle.dump(self.history.history, historyFile)
        print('history has been saved')
        
    def trainUNet(self):
        x = self.pecg.twoD.reshape((self.pecg.groups*self.pecg.length, self.pecg.height, self.pecg.width, self.pecg.channels))
        y = self.vmem.twoD.reshape((self.vmem.groups*self.vmem.length, self.vmem.height, self.vmem.width, self.vmem.channels))
        self.history = self.netg.fit(x=x, y=y, batch_size=self.batchSize, epochs=self.epochsNum, verbose=2,
                                     callbacks=self.callbacks, validation_split=self.valSplit, shuffle=True)

    def trainConvLstm(self):
        self.pecg.setRnnData(self.temporalDepth)
        self.vmem.setRnnData(self.temporalDepth)
        self.pecg.splitTrainAndVal(self.valSplit, srcType='rnn')
        self.vmem.splitTrainAndVal(self.valSplit, srcType='rnn')
        trainGenerator = utils.generatorRnn(self.pecg.train, self.vmem.train, self.batchSize)
        valGenerator = utils.generatorRnn(self.pecg.val, self.vmem.val, self.batchSize)
        self.history = self.netg.fit_generator(trainGenerator, epochs=self.epochsNum, verbose=2, callbacks=self.callbacks,
                                               validation_data=valGenerator, use_multiprocessing=True)

    def trainSeqConv5(self):
        self.history = self.netg.fit(x=self.pecg.twoD, y=self.vmem.twoD, batch_size=self.batchSize, epochs=self.epochsNum, verbose=2,
                                     callbacks=self.callbacks, validation_split=self.valSplit, shuffle=True)

    def trainUNet3d(self):
        self.history = self.netg.fit(x=self.pecg.twoD, y=self.vmem.twoD, batch_size=self.batchSize, epochs=self.epochsNum, verbose=2,
                                     callbacks=self.callbacks, validation_split=self.valSplit, shuffle=True)

    def predict(self, pecg, dstDir, trueVmem, modelDir, batchSize=16):
        self.netg.load_weights(modelDir)
        self.pecg = pecg
        self.pecg.twoD = dataProc.scale(self.pecg.twoD, self.dataRange[:2])
        predictionFunc = {
            'uNet5': self.predictUNet,
            'convLstm': self.predictConvLstm,
            'convLstm5': self.predictConvLstm,
            'uNet3d': self.predictUNet3d,
            'uNet3d5': self.predictUNet3d
        }
        predictionFunc[self.netg.name](trueVmem)
        self.prediction.twoD = dataProc.scale(self.prediction.twoD, (0, 1), dstRange=self.dataRange[2:])
        if dstDir is not None:
            self.prediction.saveData2(dstDir, 'vmem')
        mae, stdError = calculateMae(self.prediction.twoD, self.truth.twoD)
        print('mae is %f, std_error is %f'%(mae, stdError))
        return mae, stdError

    def predictConvLstm(self, trueVmem):
        self.pecg.setRnnData(self.temporalDepth)
        # Empty array to store results
        self.prediction = dataProc.Data(self.pecg.groups, self.pecg.length-self.temporalDepth+1, self.pecg.height, self.pecg.width, self.pecg.channels)
        self.truth = dataProc.Data(*self.prediction.twoD.shape)
        for i in range(0, self.pecg.groups):
            self.prediction.twoD[i] = self.netg.predict(self.pecg.rnn[i], batch_size=self.batchSize, verbose=1)
        trueVmem.setRnnData(self.temporalDepth)
        self.truth.twoD = trueVmem.rnn

    def predictUNet3d(self, trueVmem):
        self.prediction = dataProc.Data(*self.pecg.twoD.shape)
        self.prediction.twoD = self.netg.predict(self.pecg.twoD, batch_size=self.batchSize, verbose=1)
        self.truth = dataProc.Data(*self.prediction.twoD.shape)
        self.truth = trueVmem

    def predictUNet(self, trueVmem):
        x = self.pecg.twoD.reshape((self.pecg.groups*self.pecg.length, self.pecg.height, self.pecg.width, self.pecg.channels))
        y = np.zeros(x.shape, dataProc.DATA_TYPE)
        y = self.netg.predict(x, batch_size=self.batchSize, verbose=1)
        self.prediction = dataProc.Data(*self.pecg.twoD.shape)
        self.prediction.twoD = y.reshape(self.prediction.twoD.shape)
        self.truth = trueVmem


# ((number of training samples*validation split ratio)/batch size) should be an integer
'''
def trainGan(pecgDirList, memDirList, modelDir, rawSize=(200, 200), imgSize=(256, 256), channels=1, netgName='uNet', netDName='VGG16', activationG='relu', lossFuncG='mae',
temporalDepth=None, gKernels=64, gKernelSize=None, dKernels=64, dKernelSize=None, gradientPenaltyWeight=10, lossDWeight=0.01, learningRateG=1e-4, learningRateD=1e-4,
trainingRatio=5, epochsNum=100, earlyStoppingPatience=10, batchSize=10, valSplit=0.2, continueTrain=False, pretrainedGPath=None, pretrainedDPath=None, beta1=0.5, beta2=0.9):

    gan = Gan(imgSize[0], imgSize[1], channels, netDName, netgName, temporalDepth, gKernels, dKernels, gKernelSize, activationG, lossFuncG, gradientPenaltyWeight,
    lossDWeight, learningRateG, learningRateD, beta1, beta2, batchSize)
    pecg, mem, dataRange = dataProc.mergeSequence(pecgDirList, memDirList, temporalDepth, netgName, rawSize, imgSize, dataProc.NORM_RANGE)
    #delete some data here to match the batch size
    print('traing data loaded')

    trainingDataLength = math.floor((1-valSplit)*pecg.shape[0]+0.1)
    lossRecorder = np.ndarray((math.floor(trainingDataLength/gan.batchSize + 0.1)*epochsNum, 2), dtype=np.float32)
    lossCounter = 0
    minLossG = np.inf
    weightsDPath = modelDir + 'netD_latest.h5'
    weightsGPath = modelDir + 'netG_latest.h5'
    if continueTrain == True:
        gan.netG.load_weights(pretrainedGPath)
        gan.netD.load_weights(pretrainedDPath)
    labelReal = np.ones((gan.batchSize), dtype=np.float32)
    labelFake = -np.ones((gan.batchSize), dtype=np.float32)
    dummyMem = np.zeros((gan.batchSize), dtype=np.float32)
    earlyStoppingCounter = 0
    [pEcgTrain, pEcgVal, memTrain, memVal] = dataProc.splitTrainAndVal(pecg, mem, valSplit)
    print('begin to train GAN')

    for currentEpoch in range(0, epochsNum):
        beginingTime = datetime.datetime.now()
        for currentBatch in range(0, trainingDataLength, gan.batchSize):
            pEcgLocal = pEcgTrain[currentBatch:currentBatch+gan.batchSize, :]
            memLocal = memTrain[currentBatch:currentBatch+gan.batchSize, :]
            randomIndexes = np.random.randint(low=0, high=trainingDataLength-gan.batchSize-1, size=trainingRatio, dtype=np.int32)
            for i in range(0, trainingRatio):
                pEcgForD = pEcgTrain[randomIndexes[i]:randomIndexes[i]+gan.batchSize]
                memForD = memTrain[randomIndexes[i]:randomIndexes[i]+gan.batchSize]
                lossD = gan.penalizedNetD.train_on_batch([pEcgForD, memForD], [labelReal, labelFake, dummyMem])
            lossA = gan.netA.train_on_batch(pEcgLocal, [memLocal, labelReal])
        #validate the model
        lossVal = gan.netG.evaluate(x=pEcgVal, y=memVal, batch_size=gan.batchSize, verbose=0)
        lossRecorder[lossCounter, 0] = lossA[0]
        lossRecorder[lossCounter, 1] = lossVal[0]
        lossCounter += 1
        if (minLossG > lossVal[0]):
            gan.netG.save_weights(weightsGPath, overwrite=True)
            gan.netD.save_weights(weightsDPath, overwrite=True)
            minLossG = lossVal[0]
            earlyStoppingCounter = 0
        earlyStoppingCounter += 1
        displayLoss(lossD, lossA, lossVal, beginingTime, currentEpoch+1)
        if earlyStoppingCounter == earlyStoppingPatience:
            print('early stopping')
            break
    np.save(modelDir + 'loss', lossRecorder)
    print('training completed')


def prediction(pecgDirList, dstDirList, modelDir, priorEcgRange, priorMemRange, rawSize=(200, 200), imgSize=(256, 256), channels=1, netgName='uNet',
activationG='relu', lossFuncG='mae', temporalDepth=None, gKernels=64, gKernelSize=3, batchSize=10):
    network = Networks(imgSize=imgSize, channels=channels, gKernels=gKernels, gKernelSize=gKernelSize, temporalDepth=temporalDepth)
    if netgName == 'uNet':
        netG = network.uNet()
    elif netgName == 'convLstm':
        netG = network.convLstm()
    elif netgName == 'uNet3d':
        netG = network.uNet3d()
    netG.load_weights(modelDir)
    length = len(pecgDirList)
    for i in range(0, length):
        rawEcg = dataProc.loadData(srcDir=pecgDirList[i], resize=True, dstSize=imgSize)
        if netgName == 'uNet':
            ecg = rawEcg
        else:
            ecg = dataProc.create3dEcg(rawEcg, temporalDepth, netgName)
        ecg = dataProc.scale(ecg, priorEcgRange, dataProc.NORM_RANGE)
        mem = netG.predict(ecg, batch_size=batchSize, verbose=1)
        pngMem = mem*255
        mem = dataProc.scale(mem, dataProc.NORM_RANGE, priorMemRange)
        dstDir = dstDirList[i]
        if not os.path.exists(dstDir):
            os.makedirs(dstDir)
        npyDir = dstDir + 'npy/'
        if not os.path.exists(npyDir):
            os.mkdir(npyDir)
        pngDir = dstDir + 'png/'
        if not os.path.exists(pngDir):
            os.mkdir(pngDir)
        for i in range (0, mem.shape[0]):
            resizedMem = cv.resize(mem[i], rawSize)
            np.save(npyDir+'%06d.npy'%i, resizedMem)
            resizedPng = cv.resize(pngMem[i], rawSize)
            cv.imwrite(pngDir+'%06d.png'%i, resizedPng)
'''


def displayLoss(lossD, lossA, lossVal, beginingTime, epoch):
    lossValStr = ' - lossVal: ' + '%.6f'%lossVal[0]
    lossDStr = ' - lossD: ' + lossD[0].astype(np.str) + ' '
    lossDOnRealStr = ' - lossD_real: ' + lossD[1].astype(np.str) + ' '
    lossDOnFakeStr = ' - lossD_fake: ' + lossD[2].astype(np.str) + ' '
    lossDOnPenalty = ' - penalty: ' + lossD[3].astype(np.str) + ' '
    lossAStr = ' - lossA: ' + lossA[0].astype(np.str) + ' '
    lossGStr = ' - lossG: ' + lossA[1].astype(np.str) + ' '
    lossDInA = ' - lossD: ' + lossA[2].astype(np.str) + ' '
    endTime = datetime.datetime.now()
    trainingTime = endTime - beginingTime
    msg = ' - %d'%trainingTime.total_seconds() + 's' + ' - epoch: ' + '%d '%(epoch) + lossDStr + lossDOnRealStr + lossDOnFakeStr \
          + lossDOnPenalty + lossAStr + lossGStr + lossDInA + lossValStr
    print(msg)

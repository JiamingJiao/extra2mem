import keras.utils
import math
import numpy as np


class generatorRnn(keras.utils.Sequence):
    # batch generator, generate data from memory
    # generate and read from memory, or generate and save to disk then read data from disk?
    def __init__(self, pecg, vmem, batchSize):
        self.pecg = pecg  # 3D data, a view of the 2D data array
        self.vmem = vmem
        self.groups = self.pecg.shape[0]
        self.length = self.pecg.shape[1]
        self.batchSize = batchSize
        
    def __len__(self):
        return int(self.groups*self.length/self.batchSize)

    def __getitem__(self, index):
        groupIndexBegin = int(math.floor(index/self.length))
        groupIndexEnd = int(math.floor((index+self.batchSize)/self.length))
        timeIndexBegin = int(index % self.length)
        timeIndexEnd = int((index+self.batchSize) % self.length)
        if groupIndexBegin == groupIndexEnd:
            pecgBatch = self.pecg[groupIndexBegin, timeIndexBegin: timeIndexEnd]
            vmemBatch = self.vmem[groupIndexBegin, timeIndexBegin: timeIndexEnd]
        else:
            pecgBatch = np.concatenate((self.pecg[groupIndexBegin, timeIndexBegin:], self.pecg[groupIndexEnd, :timeIndexEnd]))
            vmemBatch = np.concatenate((self.vmem[groupIndexBegin, timeIndexBegin:], self.vmem[groupIndexEnd, :timeIndexEnd]))
        return (pecgBatch, vmemBatch)


class GeneratorCnn3d(keras.utils.Sequence):

    def __init__(self, pecg_paths, vmem_paths, batch_size, data_shape):
        self.pecg_paths = pecg_paths
        self.vmem_paths = vmem_paths
        self.batch_size = batch_size
        self.pecg = np.zeros((self.batch_size,)+(data_shape), np.float32)
        self.vmem = np.zeros((self.batch_size,)+(data_shape), np.float32)

    def __len__(self):
        return int(len(self.pecg_paths)/self.batch_size)

    def __getitem__(self, idx):
        block_cnt = 0
        for k in range(idx*self.batch_size, (idx+1)*self.batch_size):
            self.pecg[block_cnt] = np.load(self.pecg_paths[k])
            self.vmem[block_cnt] = np.load(self.vmem_paths[k])
            block_cnt += 1
        return (self.pecg, self.vmem)


class GeneratorBlocksToImg(keras.utils.Sequence):

    def __init__(self, pecg_paths, vmem_paths, batch_size, block_length, dst_shape):
        assert batch_size<= block_length, 'batch size > block length, cannot use this block data'
        self.pecg_paths = pecg_paths
        self.vmem_paths = vmem_paths
        self.batch_size = batch_size
        self.block_length = block_length
        self.pecg = np.zeros((self.batch_size,)+(dst_shape), np.float32)
        self.vmem = np.zeros((self.batch_size,)+(dst_shape), np.float32)

    def __len__(self):
        return len(self.pecg_paths)*int(self.block_length/self.batch_size)

    def __getitem__(self, idx):
        for k in range(idx*self.batch_size, (idx+1)*self.batch_size):
            src_file_idx = k//self.block_length
            src_pecg = np.load(self.pecg_paths[src_file_idx])
            src_vmem = np.load(self.vmem_paths[src_file_idx])
            start_idx = k%self.block_length
            end_idx = start_idx + self.batch_size
            self.pecg = src_pecg[start_idx:end_idx, ...]
            self.vmem = src_vmem[start_idx:end_idx, ...]
        return (self.pecg, self.vmem)

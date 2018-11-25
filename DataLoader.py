import torchvision
from PIL import Image
from torch.utils import data as data_utils
import glob
import os
import numpy as np
import torch

#transform = torchvision.transforms.ToTensor()


class SimulatedDataset(data_utils.Dataset):
    """
    Creates a Pytorch Dataset for ImageNet.
    """

    def __init__(self, index, root='Input_data/'):
        self.root = root
        data=torch.load(root + 'Data_{}.pth'.format(index[0]))


        length_index=len(index)
        signals = self.shorten(data['data_tensor'])
        labels = self.shorten(data['target_tensor'])
        tmp_index = self.shorten_v(data['tmp_index'])
        if length_index>1:

            for ind in index[1:]:
                data_i=torch.load(root + 'Data_{}.pth'.format(ind))
                signals=torch.cat((signals,self.shorten(data_i['data_tensor'])),0)
                labels=torch.cat((labels,self.shorten(data_i['target_tensor'])),0)
                tmp_index=np.concatenate((tmp_index, self.shorten_v(data_i['tmp_index'])),axis=0)
        #tmp_index=torch.tensor(tmp_index, dtype=torch.int)

        #data=[signals,labels,tmp_index]
        self.signals = signals
        self.labels = labels
        self.tmp_index=tmp_index
        self.n=signals.shape[0]
        #print('Labels:', labels[0:3,:])
        #print('Finished')

    def shorten(self, T):
        T1=T[1665*40:1665*41,:]
        return T1[list(range(1,1665,3)),:]

    def shorten_v(self, v):
        v1=v[1665*40:1665*41]
        return v1[list(range(1,1665,3))]


    def __len__(self):
        return self.n


    def __getitem__(self, index):

        signals=self.signals[index,:]
        labels=self.labels[index,:]
        tmp_index=self.tmp_index[index]

        return signals, labels, tmp_index


class SimulatedDataEC(data_utils.Dataset):
    """
    Creates a Pytorch Dataset for ImageNet.
    """

    def __init__(self, index, root='EC1862/ECG/'):
        self.root = root
        data=torch.load(root + 'Ecg_{}.pth'.format(index[0]))


        length_index=len(index)
        signal_full=data['Ecg']
        (_, l, t) = signal_full.shape
        signals = self.shorten(signal_full,l,t)
        labels = (data['label'])


        if length_index>1:

            for ind in index[1:]:
                data_i=torch.load(root + 'Ecg_{}.pth'.format(ind))
                signals=torch.cat((signals,self.shorten(data_i['Ecg'],l,t)),0)
                labels=torch.cat((labels,(data_i['label'])),0)


        self.signals = signals
        self.labels = labels

        self.n=signals.shape[0]
        #print('Labels:', labels[0:3,:])
        #print('Finished')

    def shorten(self, T,l,t):
        T1= T[:,list(range(1,l,2)),:]

        return T1[:,:,list(range(1,t,2))]



    def __len__(self):
        return self.n


    def __getitem__(self, index):

        signals=self.signals[index,:,:]
        labels=self.labels[index]


        return signals, labels

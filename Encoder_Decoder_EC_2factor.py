from __future__ import print_function
import argparse
import torch
#import torch.utils.data
import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

import torch.utils.data as data_utils
import uuid
import os
import scipy.io as sio
import numpy as np
import random
import math
from torch import nn, optim
from torch.autograd import Variable
#from dataLoader import SimulatedDataset
from DataLoader import SimulatedDataEC,SimulatedDataEC_2factor

import matplotlib.pyplot as plt

from lossDefinition import loss_function, multip, loss_function_deterministic
from modelDefinition import svsVAE, svsVAE_deterministic, svsLanguage, svsLanguage_classic, svsLanguage_classic_deterministic, svsLanguage_deterministic, sssVAE

parser = argparse.ArgumentParser(description='Bayesian TransCoder')
parser.add_argument('--batchsize', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 10)')
parser.add_argument('--node_dim', type=int, default=1862, metavar='N',
                    help='Dimension of TMP')
parser.add_argument('--lead_dim', type=int, default=120, metavar='N',
                    help='Dimension of ECG lead')
parser.add_argument('--time_dim', type=int, default=201, metavar='N',
                    help='time dimension')
parser.add_argument('--epochs', type=int, default=16, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--epoch_start', type=int, default=0, metavar='N',
                    help='start num for epoch')
parser.add_argument('--train_from', type=str, default=None, metavar='N',
                    help='model to train from')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--architecture', type=str, default='svs', metavar='N',
                    help='architecture could be sss, svs or classicLang')
parser.add_argument('--input_type', type=str, default='y', metavar='N',
                    help='input type can be y or u')
parser.add_argument('--latent_type', type=str, default='stochastic', metavar='N',
                    help='either stochastic or deterministic')
parser.add_argument('--validation_style', type=str, default='validation_exc', metavar='N',
                    help='validation_exc or validation_infarct')
parser.add_argument('--dataset', type=str, default='EC1862', metavar='N',
                    help='EC1862 or DC2144 or AW1898')
parser.add_argument('--path', type=str, default='EC1862/', metavar='N',
                    help='path to dataset')
parser.add_argument('--lr', type=float, default=1e-4, metavar='N',
                    help='Learning rate')
parser.add_argument('--beta', type=float, default=1, metavar='N',
                    help='beta factor controlling two mutual information terms')
parser.add_argument('--decay', type=float, default=1.0, metavar='N',
                    help='decay rate')

parser.add_argument('--uuid', type=str, default=uuid.uuid1(), help='(somewhat) unique identifier for the model/job')
parser.add_argument('--is_dropout', type=int, default=0, metavar='N',
                    help='This is latent dropout which can be True or False')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
#torch.manual_seed(args.seed)

#segment_size = 17
#dataSampleInterval = 50
isAnnealing = 0


architecture=args.architecture
input_type=args.input_type
vae_type=args.latent_type
validation_style=args.validation_style
is_dropout=args.is_dropout
#batch_size = args.batchsize
#dataset=args.dataset


if not os.path.isdir('crash_recovery'):
    os.mkdir('crash_recovery')
if not os.path.isdir('Output'):
    os.mkdir('Output')
if not os.path.isdir('Plots'):
    os.mkdir('Plots')

if not os.path.isdir(os.path.join('Plots',str(args.uuid))):
    os.mkdir(os.path.join('Plots',str(args.uuid)))
#infarct_loc_for_validation=11

#LAMBDA_GP=0.05


if args.cuda:
    torch.cuda.manual_seed(args.seed)



def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def checkAberrant(U):
    actTime = (U > 0.4).sum(1)
    numHealthy = sum(actTime > 60)
    if numHealthy < 200:
        return 0
    else:
        return 1

def plotTMP(x_r, x, epoch):
    i=np.random.randint(x.size(1))
    x_true=x[:,i,:]
    x=x_r[:,i,:]

    plt.subplot(311)
    plt.plot(x[:,300:310].numpy())
    plt.subplot(312)
    plt.plot(x_true[:,300:305].numpy())

    plt.subplot(313)
    plt.plot(x_true[:, 305:310].numpy())

    plt.savefig(os.path.join('Plots', str(args.uuid), str(epoch) + ".png"))
    plt.close()

def restructure(Y):
    # This is batch by Long vector
    (B,N,T)=Y.size()

    ####Plot starts------------
    # x = Y[1, :, :]
    #
    # plt.figure(1)
    # plt.subplot(211)
    # plt.plot(x[20:50, :].transpose(0,1).numpy())
    # plt.show()
    #

    ##Plot ends

    return Y.permute(2,0,1)



def readBatch(index, path):
    index=list(index.numpy())
    for ind in index:
        TrainData = sio.loadmat(path + 'Tmp' + str(ind) + '.mat')
        Ut = TrainData['U']
        #(a, b) = Ut.shape
        #Ut = np.reshape(Ut, (b, a), order='F')
        (a, b) = Ut.shape

        # print('The size of U is: ', a,',',b)
        if a == args.node_dim:
            U_np = Ut.transpose()

        elif b == args.node_dim:
            U_np = Ut

        else:
            print('The TMP matrix does not have proper dimension')

        discrete_time=range(1,args.time_dim,2)
        U_np=U_np[discrete_time,:]
        seq_length=U_np.shape[0]
        #discrete_node=range(0,args.node_dim,3)

        ####Plot starts------------

        # plt.figure(1)
        # plt.subplot(211)
        # plt.plot(U_np[:, 50:60])
        # plt.show()

        ##Plot ends

        zz = torch.FloatTensor(U_np)
        U = zz.contiguous().view(seq_length, 1, -1)

        if 'outData' in locals():
            outData = torch.cat((outData, U), 1)
        else:
            outData = U


    return outData

def train(epoch, train_loader, model, optimizer):
    model.train()
    train_loss = 0
    pathTmp=args.path+'TMP/'
    N=len(train_loader.dataset)

    for batch_index, (outY, label) in enumerate(train_loader):
        #tmp_idx=tmp_index.data
        outData = readBatch(label, pathTmp)
        outY = restructure(outY)
        data = Variable(outData)  # sequence length, batch size, input size
        Y = Variable(outY)


        if isAnnealing:
            if epoch < 50:
                annealParam = 0
            elif epoch < 500:
                annealParam = (epoch / 500)
            else:
                annealParam = 1
        else:
            annealParam = 1
        annealParam = Variable(torch.FloatTensor([annealParam]))
        if args.cuda:
            data = data.cuda()

            Y =Y.cuda()

            #elif input_type == 'u':
            annealParam=annealParam.cuda()


        optimizer.zero_grad()



        if input_type == 'y':
            if vae_type == 'deterministic':
                muTheta, mu = model(Y)
            elif vae_type == 'stochastic':
                muTheta, logvarTheta, mu, logvar = model(Y)

        elif input_type == 'u':
            if vae_type == 'deterministic':
                muTheta, mu = model(data)
            elif vae_type == 'stochastic':
                muTheta, logvarTheta, mu, logvar = model(data)


        if vae_type == 'deterministic':
            loss = loss_function_deterministic(muTheta, data, args)

        elif vae_type == 'stochastic':


            loss = loss_function(muTheta, logvarTheta, data, mu, logvar, annealParam, args)



        # print(loss)

        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()

        #if batch_index % 120==0:
        #    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_index, N,100. * batch_index*args.batchsize / N,loss.data[0]))
        # j=j+dataSampleInterval
    train_loss_avg=train_loss*args.batchsize/N
    print('====>Train Epoch: {} Average Train loss: {:.4f}'.format(
        epoch, train_loss_avg))
    return torch.FloatTensor([train_loss_avg])



def test(epoch, test_loader, model, optimizer):
    model.eval()

    test_loss = 0
    pathTmp=args.path+'TMP/'
    N=len(test_loader.dataset)

    for batch_index, (outY, tmp_index) in enumerate(test_loader):
        outData = readBatch(tmp_index, pathTmp)
        outY=restructure(outY)
        data = Variable(outData)  # sequence length, batch size, input size
        Y = Variable(outY)

        optimizer.zero_grad()

        if args.cuda:
            data = data.cuda()
            if input_type == 'y_projected':
                Y_projected=Y_projected.cuda()
            elif input_type == 'y':
                Y=Y.cuda()

        if input_type == 'y_projected':  # or u or y
            if vae_type == 'deterministic':
                muTheta, mu = model(Y_projected)
            elif vae_type == 'stochastic':
                muTheta, logvarTheta, _ ,_ = model(Y_projected)

        elif input_type == 'y':
            if vae_type == 'deterministic':
                muTheta, mu = model(Y)
            elif vae_type == 'stochastic':
                muTheta, logvarTheta, _ ,_ = model(Y)

        elif input_type == 'u':
            if vae_type == 'deterministic':
                muTheta, mu = model(data)
            elif vae_type == 'stochastic':
                muTheta, logvarTheta, _ ,_ = model(data)


        #muTheta, logvarTheta, _, _ = model(Y_projected)
        loss = loss_function_deterministic(muTheta, data, args)

        #loss = torch.sum((muTheta - data).pow(2)) / (201*batch_size*1862)

        test_loss += loss.data[0]
        #print('Test Epoch: {}, batch:{}'.format(epoch,j))

    if epoch % 50 == 0:
        plotTMP(muTheta.cpu().data, data.cpu().data, epoch)

    avg_error=test_loss *args.batchsize/ N
    print('====> Test Epoch: {} Average Test loss: {:.4f}'.format(
        epoch, avg_error ))
    return torch.FloatTensor([avg_error])


def main():
    print('batch size:', args.batchsize)

    #ECG_dim = 120
    latent_dim = 12
    TMP_dim=1862
    timestep=100
    #print('segment size:', segment_size)

    input_dim=args.lead_dim // 2
    mid_input=30

    trainData=SimulatedDataEC_2factor(list(range(4,8)),list(range(5,8)))
    #N_train=trainData.len()

    train_loader = data_utils.DataLoader(trainData, batch_size=args.batchsize,
                                         shuffle=True)
    testData=SimulatedDataEC_2factor([1],[6])
    #N_test=testData.len()
    test_loader = data_utils.DataLoader(testData, batch_size=args.batchsize,
                                       shuffle=True)


    if architecture == 'svs':
        if vae_type=='deterministic':
            model = svsVAE_deterministic(input_dim, TMP_dim, timestep, mid_input, 800, latent_dim, 800, 40)
        elif vae_type =='stochastic':
            model = svsVAE(input_dim, TMP_dim, timestep, mid_input, 800, latent_dim, 800, 40)


    elif architecture == 'classicLang':
        if vae_type=='deterministic':
            model = svsLanguage_classic_deterministic(input_dim, TMP_dim, timestep, mid_input, 800, latent_dim)
        elif vae_type =='stochastic':
            model = svsLanguage_classic(input_dim, TMP_dim, timestep, mid_input, 800, latent_dim)



    elif architecture == 'svsLang':
        if vae_type == 'deterministic':
            model = svsLanguage_deterministic(input_dim, TMP_dim, timestep, mid_input, 800, latent_dim, 800, 40)
        elif vae_type == 'stochastic':
            model = svsLanguage(input_dim, TMP_dim, timestep, mid_input, 800, latent_dim, 800, 40)


    else:
        print('Wrong architecture')


    if args.cuda:
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    savefile='EC'+vae_type + '_' + architecture +'i_4_7_beta'+str(args.beta)


    if args.train_from is not None:

        print("=> loading checkpoint '{}'".format(args.train_from))
        checkpoint = torch.load(args.train_from)
        args.epoch_start = checkpoint['epoch']

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.train_from, checkpoint['epoch']))
    try:
        lr = args.lr
        #start = time.time()
        train_error = np.zeros(args.epochs)
        #validation_scores = np.zeros(args.epochs)
        test_error = np.zeros(args.epochs)
        for epoch in range(args.epoch_start, args.epochs):

            train_error[epoch] = train(epoch, train_loader, model, optimizer)
            if (epoch % 250 == 0):
                save_checkpoint({
                    'args':args,
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                },
                    'Output/' + savefile+ '_' + str(epoch))

            test_error[epoch] = test(epoch, test_loader, model, optimizer)
            # if (epoch % 10 == 0):
            # test_error = torch.FloatTensor([test_error])

            if epoch == args.epoch_start:

                min_err = test_error[args.epoch_start]
            else:

                if epoch > 400 and (test_error[epoch] < min_err):
                    min_err = test_error[epoch]
                    save_checkpoint({
                        'args': args,
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    },
                        'Output/' +savefile + '_min_err')

            lr = lr * args.decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr



        save_checkpoint({
            'trainError': train_error,
            'testError': test_error,
        },
            'Output/testArr_' + savefile)




    except KeyboardInterrupt:
        save_checkpoint({
            'args': args,
            'epoch': epoch,

            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),

            'crash': 'Keyboard interrupt'
        }, filename=os.path.join('crash_recovery', savefile))
        print("Keyboard Interrupt")
        exit(0)

    except Exception as e:
        save_checkpoint({
            'args': args,
            'epoch': epoch ,

            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),

            'crash': e
        }, filename=os.path.join('crash_recovery', savefile))
        raise e



if __name__ == '__main__':
    main()

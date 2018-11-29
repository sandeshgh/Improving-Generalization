from __future__ import print_function
import argparse
import torch
import torch.utils.data
import os
import numpy as np
import random
import scipy.io as sio
from torch.autograd import Variable
import matplotlib.pyplot as plt
import sys
import numpy as np
from numpy import array
import scipy as sp
import scipy.stats as stats
from modelDefinition import svsVAE, sssVAE, svsVAE_deterministic, svsLanguage_classic, svsLanguage_classic_deterministic


#from SVSTranscoder_single_task import readH, genNoisy,checkAberrant

methods=[1,0,0,0]

test_size=100

TMP_dim = 2097
latent_dim = 12

timestep=123
    #print('segment size:', segment_size)

input_dim=12
mid_input=12


def calc_MSE(u,M):
    mse=np.sum(np.square(u-M))/np.prod(u.shape)
    return mse

def calc_corr_Pot(u,x):
    w,m,n=u.shape
    correlation_sum=0
    for i in range(m):
        for j in range(n):
            correlation_sum=correlation_sum+ stats.pearsonr(u[:,i,j], x[:,i,j])[0]
    return(correlation_sum/(m*n))

def calc_AT(u,M):
    w, m, n = u.shape
    u_new=np.roll(u,-1,axis=0)
    u_slope=(u_new-u)[:-1,:,:]
    u_AT=np.argmax(u_slope,axis=0)

    M_new=np.roll(M,-1,axis=0)
    M_slope=(M_new-M)[:-1,:,:]
    M_AT=np.argmax(M_slope,axis=0)

    corr_coeff=0
    #print('Size of AT is :', u_AT.shape)
    u_apd = np.sum((u > 0.7), axis=0)
    u_scar = u_apd < (0.25 * w)
    x_apd = np.sum((M > 0.7), axis=0)
    x_scar = x_apd < (0.25 * w)

    u_AT[u_scar]=200
    M_AT[x_scar]=200


    #m,n=u_AT.shape
    for i in range(m):
        true_AT=u_AT[i,:]
        x_AT=M_AT[i,:]
        corr_coeff = corr_coeff + stats.pearsonr(true_AT, x_AT)[0]

    return corr_coeff/m

def calc_DC(u,x):
    w,m,n=u.shape
    u_apd=np.sum((u>0.7),axis=0)
    u_scar=u_apd>0.25*w
    x_apd=np.sum((x>0.7),axis=0)
    x_scar=x_apd>0.25*w
    dice_coeff=0

    #print('Size of u_scar is :', u_scar.shape)

    for i in range(m):
        u_row=u_scar[i,:]
        x_row=x_scar[i,:]
        u_scar_index=np.where(u_row==0)[0]
        x_scar_index=np.where(x_row==0)[0]

        intersect=set(u_scar_index) & set(x_scar_index)

        dice_coeff=dice_coeff+len(intersect)/float(len(set(u_scar_index))+len(set(x_scar_index)))

    return 2*dice_coeff/m, u_scar, x_scar





model_v=svsVAE(input_dim, TMP_dim, timestep, mid_input, 800, latent_dim, 800, 40)
#model_l=svsVAE_deterministic(120, TMP_dim,201, 60, 800, latent_dim,800,40)
model_l=svsLanguage_classic(input_dim, TMP_dim, timestep, mid_input, 800, latent_dim)





#modelfull_v800 = torch.load('TranscoderOutput/stochastic_svsLang_modely_validation_infarct980', map_location={'cuda:0': 'cpu'})
modelfull_v= torch.load('OutputSmall/stochastic_svs_y_validation_exc_dropout_True_min_err', map_location={'cuda:0': 'cpu'})
model_v.load_state_dict(modelfull_v['state_dict'])
model_v.eval()

#modelfull_no_dropout=

model_v_determ=svsVAE_deterministic(input_dim, TMP_dim, timestep, mid_input, 800, latent_dim, 800, 40)
modelfull_v= torch.load('OutputSmall/deterministic_svs_y_validation_exc_dropout_True_min_err', map_location={'cuda:0': 'cpu'})
model_v_determ.load_state_dict(modelfull_v['state_dict'])
model_v_determ.eval()

#modelfull_v1000 = torch.load('TranscoderOutput/stochastic_svsLang_modely_validation_infarct980', map_location={'cuda:0': 'cpu'})
modelfull_l = torch.load('OutputSmall/stochastic_classicLang_y_validation_exc_dropout_True_min_err', map_location={'cuda:0': 'cpu'})
model_l.load_state_dict(modelfull_l['state_dict'])
model_l.eval()

model_l_determ=svsLanguage_classic_deterministic(input_dim, TMP_dim, timestep, mid_input, 800, latent_dim)
modelfull_l = torch.load('OutputSmall/deterministic_classicLang_y_validation_exc_dropout_True_min_err', map_location={'cuda:0': 'cpu'})
model_l_determ.load_state_dict(modelfull_l['state_dict'])
model_l_determ.eval()


testData=SimulatedDataset([8])
    #N_test=testData.len()
test_loader = data_utils.DataLoader(testData, batch_size=args.batchsize,
                                       shuffle=True)

MSE_svs=[]
Corr_Pot_svs=[]
Corr_AT_svs=[]
DC_svs=[]

MSE_Language=[]
Corr_Pot_Language=[]
Corr_AT_Language=[]
DC_Language=[]

MSE_svs_determ=[]
Corr_Pot_svs_determ=[]
Corr_AT_svs_determ=[]
DC_svs_determ=[]

MSE_Language_determ=[]
Corr_Pot_Language_determ=[]
Corr_AT_Language_determ=[]
DC_Language_determ=[]

for batch_index, (outY, label, tmp_index) in enumerate(test_loader):
    outData = readBatch(tmp_index, pathTmp)
    outY = restructure(outY)
    outData = Variable(outData)  # sequence length, batch size, input size
    outY = Variable(outY)

    if args.cuda:
        outData = outData.cuda()

        outY = outY.cuda()

    u = outData.data.numpy()

    if methods[0]:
        muTheta_v, _, mu_v, logvar_v = model_v((outY))
        M_v=muTheta_v.data.numpy()
        MSE_svs.append(calc_MSE(u, M_v))
        Corr_Pot_svs.append(calc_corr_Pot(u, M_v))
        Corr_AT_svs.append(calc_AT(u, M_v))
        DC_svs.append(calc_AT(u, M_v))
    #print('mu,logvar greater than 1',torch.sum(mu_v>1),torch.sum(torch.exp(logvar_v)>1))
    if methods[1]:
        muTheta_l, _, mu_l, logvar_l = model_l((outY))
        M_l = muTheta_l.data.numpy()

        MSE_Language.append(calc_MSE(u, M_l))
        Corr_Pot_Language.append(calc_corr_Pot(u, M_l))
        Corr_AT_Language.append(calc_AT(u, M_l))
        DC_Language.append(calc_AT(u, M_l))

    if methods[2]:


        muTheta_v_determ, mu_v_determ = model_v_determ((outY))
        M_v_determ = muTheta_v_determ.data.numpy()

        MSE_svs_determ.append(calc_MSE(u, M_v_determ))
        Corr_Pot_svs_determ.append(calc_corr_Pot(u, M_v_determ))
        Corr_AT_svs_determ.append(calc_AT(u, M_v_determ))
        DC_svs_determ.append(calc_AT(u, M_v_determ))

    #print('mu,logvar greater than 1',torch.sum(mu_l>1),torch.sum(torch.exp(logvar_l)>1))
    if methods[3]:
        muTheta_l_determ, mu_l_determ= model_l_determ((outY))
        M_l_determ = muTheta_l_determ.data.numpy()

        MSE_Language_determ.append(calc_MSE(u, M_l_determ))
        Corr_Pot_Language_determ.append(calc_corr_Pot(u, M_l_determ))
        Corr_AT_Language_determ.append(calc_AT(u, M_l_determ))
        DC_Language_determ.append(calc_AT(u, M_l_determ))






if methods[0]: print('Mean square error, svs ; mean : {}, std:{}'.format(np.array(MSE_svs).mean(), np.array(MSE_svs).std()))
if methods[2]: print('Mean square error, Language ; mean : {}, std:{}'.format(np.array(MSE_Language).mean(), np.array(MSE_Language).std()))
if methods[1]: print('Mean square error, svs deterministic ; mean : {}, std:{}'.format(np.array(MSE_svs_determ).mean(), np.array(MSE_svs_determ).std()))
if methods[3]: print('Mean square error, Language Deterministic:'.format(np.array(MSE_Language_determ).mean(), np.array(MSE_Language_determ).std()))


if methods[0]: print('Correlation of potential,svs; mean : {}, std:{}'.format(np.array(Corr_Pot_svs).mean(), np.array(Corr_Pot_svs).std()))
if methods[2]: print('Correlation of potential,Language; mean : {}, std:{}'.format(np.array(Corr_Pot_Language).mean(), np.array(Corr_Pot_Language).std()))
if methods[1]: print('Correlation of potential,svs deterministic; mean : {}, std:{}'.format(np.array(Corr_Pot_svs_determ).mean(), np.array(Corr_Pot_svs_determ).std()))
if methods[3]: print('Correlation of potential,Language, Deterministic; mean : {}, std:{}'.format(np.array(Corr_Pot_Language_determ).mean(), np.array(Corr_Pot_Language_determ).std()))


if methods[0]: print('Correlation of AT,svs; mean : {}, std:{}'.format(np.array(Corr_AT_svs).mean(), np.array(Corr_AT_svs).std()))
if methods[2]: print('Correlation of AT,Language; mean : {}, std:{}'.format(np.array(Corr_AT_Language).mean(), np.array(Corr_AT_Language).std()))
if methods[1]: print('Correlation of AT,svs deterministic; mean : {}, std:{}'.format(np.array(Corr_AT_svs_determ).mean(), np.array(Corr_AT_svs_determ).std()))
if methods[3]: print('Correlation of AT,Language, Deterministic; mean : {}, std:{}'.format(np.array(Corr_AT_Language_determ).mean(), np.array(Corr_AT_Language_determ).std()))

if methods[0]: print('DC, svs ; mean : {}, std:{}'.format(np.array(DC_svs).mean(), np.array(DC_svs).std()))
if methods[2]: print('DC, Language ; mean : {}, std:{}'.format(np.array(DC_Language).mean(), np.array(DC_Language).std()))
if methods[1]: print('DC, svs deterministic ; mean : {}, std:{}'.format(np.array(DC_svs_determ).mean(), np.array(DC_svs_determ).std()))
if methods[3]: print('DC, Language Deterministic:'.format(np.array(DC_Language_determ).mean(), np.array(DC_Language_determ).std()))

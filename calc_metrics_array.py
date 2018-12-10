from __future__ import print_function
import argparse
import torch
import torch.utils.data
import torch.utils.data as data_utils
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
from DataLoader import SimulatedDataEC, SimulatedDataEC_2factor
from Encoder_Decoder_EC import readBatch, restructure


#from SVSTranscoder_single_task import readH, genNoisy,checkAberrant
isCuda=torch.cuda.is_available()
methods=[1,1,0,0]

test_size=10

TMP_dim = 1862
latent_dim = 12


timestep=100
    #print('segment size:', segment_size)

input_dim=60
mid_input=30


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

    return 2*dice_coeff/m #, u_scar, x_scar


model_v = svsVAE(input_dim, TMP_dim, timestep, mid_input, 800, latent_dim, 800, 40)

model_v_determ = svsVAE_deterministic(input_dim, TMP_dim, timestep, mid_input, 800, latent_dim, 800, 40)



#modelfull_v= torch.load('Output/ECstochastic_svsi_4_7_beta1_min_err', map_location={'cuda:0': 'cpu'})
#modelfull_v= torch.load('Output/ECstochastic_svs_y_18_22_min_err', map_location={'cuda:0': 'cpu'})
modelfull_v= torch.load('Output/ECstochastic_svsi_3_9_beta1_min_err', map_location={'cuda:0': 'cpu'})
#modelfull_v= torch.load('Output/ECstochastic_svs_y_min_err', map_location={'cuda:0': 'cpu'})
model_v.load_state_dict(modelfull_v['state_dict'])
model_v.eval()

#modelfull_v= torch.load('Output/ECdeterministic_svsi_4_7_beta1_min_err', map_location={'cuda:0': 'cpu'})
#modelfull_v= torch.load('Output/ECdeterministic_svs_y_18_22_min_err', map_location={'cuda:0': 'cpu'})
modelfull_v= torch.load('Output/ECdeterministic_svsi_3_9_beta1_min_err', map_location={'cuda:0': 'cpu'})
#modelfull_v= torch.load('Output/ECdeterministic_svs_y_min_err', map_location={'cuda:0': 'cpu'})
model_v_determ.load_state_dict(modelfull_v['state_dict'])
model_v_determ.eval()


MSEarr_sto_mean=np.array([])
MSEarr_sto_std=np.array([])

MSEarr_det_mean=np.array([])
MSEarr_det_std=np.array([])

CorrPotarr_sto_mean=np.array([])
CorrPotarr_sto_std=np.array([])

CorrPotarr_det_mean=np.array([])
CorrPotarr_det_std=np.array([])

CorrATarr_sto_mean=np.array([])
CorrATarr_sto_std=np.array([])

CorrATarr_det_mean=np.array([])
CorrATarr_det_std=np.array([])

DCarr_sto_mean=np.array([])
DCarr_sto_std=np.array([])

DCarr_det_mean=np.array([])
DCarr_det_std=np.array([])

#testRange=np.append(np.arange(1,4), np.arange(8,11))
#testRange=np.append(np.arange(1,18), np.arange(23,41))
testRange=np.append(np.arange(1,3), np.arange(10,12))
#testRange=[10,35]

for test_index in (testRange):
    print('Test index:',test_index)
    testData=SimulatedDataEC_2factor([test_index],[6])
    #testData = SimulatedDataEC([test_index])


    test_loader = data_utils.DataLoader(testData, batch_size=test_size,
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

    pathTmp='EC1862/TMP/'

    for batch_index, (outY, tmp_index) in enumerate(test_loader):
        outData = readBatch(tmp_index, pathTmp)
        outY = restructure(outY)
        data = Variable(outData)  # sequence length, batch size, input size
        outY = Variable(outY)


        # if isCuda:
        #     outData = outData.cuda()
        #
        #     outY = outY.cuda()

        u = outData.numpy()

        if methods[0]:
            muTheta_v, _, mu_v, logvar_v = model_v((outY))
            M_v=muTheta_v.data.numpy()
            MSE_svs.append(calc_MSE(u, M_v))
            Corr_Pot_svs.append(calc_corr_Pot(u, M_v))
            Corr_AT_svs.append(calc_AT(u, M_v))
            DC_svs.append(calc_DC(u, M_v))
        #print('mu,logvar greater than 1',torch.sum(mu_v>1),torch.sum(torch.exp(logvar_v)>1))
        if methods[2]:
            muTheta_l, _, mu_l, logvar_l = model_l((outY))
            M_l = muTheta_l.data.numpy()

            MSE_Language.append(calc_MSE(u, M_l))
            Corr_Pot_Language.append(calc_corr_Pot(u, M_l))
            Corr_AT_Language.append(calc_AT(u, M_l))
            DC_Language.append(calc_DC(u, M_l))

        if methods[1]:


            muTheta_v_determ, mu_v_determ = model_v_determ((outY))
            M_v_determ = muTheta_v_determ.data.numpy()

            MSE_svs_determ.append(calc_MSE(u, M_v_determ))
            Corr_Pot_svs_determ.append(calc_corr_Pot(u, M_v_determ))
            Corr_AT_svs_determ.append(calc_AT(u, M_v_determ))
            DC_svs_determ.append(calc_DC(u, M_v_determ))

        #print('mu,logvar greater than 1',torch.sum(mu_l>1),torch.sum(torch.exp(logvar_l)>1))
        if methods[3]:
            muTheta_l_determ, mu_l_determ= model_l_determ((outY))
            M_l_determ = muTheta_l_determ.data.numpy()

            MSE_Language_determ.append(calc_MSE(u, M_l_determ))
            Corr_Pot_Language_determ.append(calc_corr_Pot(u, M_l_determ))
            Corr_AT_Language_determ.append(calc_AT(u, M_l_determ))
            DC_Language_determ.append(calc_DC(u, M_l_determ))

    MSEarr_sto_mean=np.append(MSEarr_sto_mean, np.array(MSE_svs).mean())
    MSEarr_sto_std=np.append(MSEarr_sto_std, np.array(MSE_svs).std())

    MSEarr_det_mean=np.append(MSEarr_det_mean, np.array(MSE_svs_determ).mean())
    MSEarr_det_std=np.append(MSEarr_det_std, np.array(MSE_svs_determ).std())

    CorrPotarr_sto_mean=np.append(CorrPotarr_sto_mean, np.array(Corr_Pot_svs).mean())
    CorrPotarr_sto_std=np.append(CorrPotarr_sto_std, np.array(Corr_Pot_svs).std())

    CorrPotarr_det_mean=np.append(CorrPotarr_det_mean, np.array(Corr_Pot_svs_determ).mean())
    CorrPotarr_det_std=np.append(CorrPotarr_det_std, np.array(Corr_Pot_svs_determ).std())

    CorrATarr_sto_mean=np.append(CorrATarr_sto_mean, np.array(Corr_AT_svs).mean())
    CorrATarr_sto_std=np.append(CorrATarr_sto_std, np.array(Corr_AT_svs).std())

    CorrATarr_det_mean=np.append( CorrATarr_det_mean, np.array(Corr_AT_svs_determ).mean())
    CorrATarr_det_std=np.append(CorrATarr_det_std, np.array(Corr_AT_svs_determ).std())

    DCarr_sto_mean=np.append(DCarr_sto_mean, np.array(DC_svs).mean())
    DCarr_sto_std=np.append(DCarr_sto_std, np.array(DC_svs).std())

    DCarr_det_mean=np.append(DCarr_det_mean, np.array(DC_svs_determ).mean())
    DCarr_det_std=np.append(DCarr_det_std, np.array(DC_svs_determ).std())

Mean_stack=np.vstack((MSEarr_sto_mean,MSEarr_det_mean, CorrPotarr_sto_mean,CorrPotarr_det_mean, CorrATarr_sto_mean, CorrATarr_det_mean, DCarr_sto_mean, DCarr_det_mean))

Std_stack=np.vstack((MSEarr_sto_std,MSEarr_det_std, CorrPotarr_sto_std,CorrPotarr_det_std, CorrATarr_sto_std, CorrATarr_det_std, DCarr_sto_std, DCarr_det_std))
np.save('Mean_metric_2factor_3_9.npy',Mean_stack)
np.save('Std_metric_2factor_3_9.npy',Std_stack)

#np.save('Mean_metric_all_18_22.npy',Mean_stack)
#np.save('Std_metric_all_18_22.npy',Std_stack)

print('Mean stack:', Mean_stack)
print('Std stack:', Std_stack)

fig=plt.figure(1)

xaxis=np.array(range(1,np.size(Mean_stack,1)+1))
ax1=fig.add_subplot(221)
ax1.errorbar(xaxis, MSEarr_sto_mean, yerr = MSEarr_sto_std, color="b", label='Stochastic')
ax1.errorbar(xaxis, MSEarr_det_mean, yerr = MSEarr_det_std, color="r", label='Deterministic')

ax2=fig.add_subplot(222)
ax2.errorbar(xaxis, CorrPotarr_sto_mean, yerr = CorrPotarr_sto_std, color="b", label='Stochastic')
ax2.errorbar(xaxis, CorrPotarr_det_mean, yerr = CorrPotarr_det_std, color="r", label='Deterministic')

ax3=fig.add_subplot(223)
ax3.errorbar(xaxis, CorrATarr_sto_mean, yerr = CorrPotarr_sto_std, color="b", label='Stochastic')
ax3.errorbar(xaxis, CorrATarr_det_mean, yerr = CorrPotarr_det_std, color="r", label='Deterministic')

ax4=fig.add_subplot(224)
ax4.errorbar(xaxis, DCarr_sto_mean, yerr = DCarr_sto_std, color="b", label='Stochastic')
ax4.errorbar(xaxis, DCarr_det_mean, yerr = DCarr_det_std, color="r", label='Deterministic')

plt.savefig('Fig_2factor_3_9')
plt.show()
plt.close()



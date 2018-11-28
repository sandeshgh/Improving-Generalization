from __future__ import print_function
import argparse
import torch
import torch.utils.data
import numpy as np
import os
import scipy.io as sio
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable

def multip(A,Tens):
    (T,B,N)=Tens.size()
    out=torch.bmm(A.repeat(B,1,1),Tens.permute(1,2,0))
    return out.permute(2,0,1)


class BayesPosFunction(Function):
    @staticmethod
    def forward(ctx, Mu, Var, beta, Y, H, U,isCuda):
        # print(Mu,Var,beta,Y,H,U)
        (N, T, B) = Var.size()
        (M, T, B) = Y.size()
        invVar = torch.reciprocal(Var)
        hMu = multip(H, Mu)
        # hMu=torch.bmm(H.repeat(B,1,1),Mu.permute(2,0,1)).permute(1,2,0)
        dU = U - Mu
        dY = Y - hMu
        logdetY = torch.FloatTensor([0])
        invSigmaYdY = (torch.zeros(M, T, B))
        Id=torch.eye(M)
        if isCuda:
            Id=Id.cuda()
            invSigmaYdY=invSigmaYdY.cuda()
            logdetY=logdetY.cuda()



        for j in range(T):
            for k in range(B):
                SigmaY = (1 / beta) * Id + torch.mm((H * Var[:, j, k]), H.transpose(0, 1))
                logdetY = logdetY + torch.sum(torch.log(torch.eig(SigmaY, eigenvectors=False)[0][:, 0]))  # Reached here
                invSigmaY = torch.inverse(SigmaY)
                invSigmaYdY[:, j, k] = torch.mv(invSigmaY, dY[:, j, k])

        invVardU = torch.mul(invVar, dU)
        #print('Var is: ', Var)
        #print('logdetY:', logdetY)

        ctx.save_for_backward(Mu, Var, beta, Y, H, U)
        cost2 = logdetY + torch.sum(torch.mul(dY, invSigmaYdY)) - torch.sum(torch.log(Var)) - torch.sum(
            torch.mul(dU, invVardU))
        logtwopi=torch.log(torch.FloatTensor([2*3.14]))
        if isCuda:
            logtwopi=logtwopi.cuda()
        cost1 = M * T * B * torch.log(beta) - N * T * B*logtwopi  - beta * torch.sum((Y - multip(H, U)).pow(2))

        cost = 0.5 * (cost1 + cost2)
        return cost / (T * N * B)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        # grad_input=grad_output.clone()
        #print('Reached inside bayes backward')
        Mu, Var, beta, Y, H, U = ctx.saved_tensors
        isCuda=U.is_cuda

        (N, T, B) = Var.size()
        (M, T, B) = Y.size()
        invVar = torch.reciprocal(Var)
        hMu = multip(H, Mu)
        #print('Reached after hMu')
        # hMu=torch.bmm(H.repeat(B,1,1),Mu.permute(2,0,1)).permute(1,2,0)
        dU = U - Mu
        dY = Y - hMu
        HSigmaYH = (torch.zeros(N, T, B))
        logdetY = torch.FloatTensor([0])
        invSigmaYdY = (torch.zeros(M, T, B))
        Id = torch.eye(M)
        if isCuda==1:
            Id=Id.cuda()
            invSigmaYdY=invSigmaYdY.cuda()
            logdetY=logdetY.cuda()
            HSigmaYH = HSigmaYH.cuda()

        for j in range(T):
            for k in range(B):
                SigmaY = (1 / beta) * Id + torch.mm((H * Var[:, j, k]), H.transpose(0, 1))
                logdetY = logdetY + torch.sum(torch.log(torch.eig(SigmaY, eigenvectors=False)[0][:, 0]))  # Reached here
                invSigmaY = torch.inverse(SigmaY)
                invSigmaYdY[:, j, k] = torch.mv(invSigmaY, dY[:, j, k])
                HSigmaYH[:, j, k] = torch.diag(torch.mm(torch.mm(H.transpose(0, 1), invSigmaY), H))
        #print('Reached after for loop')
        invVardU = torch.mul(invVar, dU)
        #print('Reached invVardU')
        HSigmadY = multip(H.transpose(0, 1), invSigmaYdY)

        gradMu = invVardU - HSigmadY
        gradSigmaU = invVardU.pow(2) - invVar + HSigmaYH - HSigmadY.pow(2)
        gradBeta = 0.5 * (M * T * B / beta - torch.sum((Y - multip(H, U)).pow(2)) + (1 / beta.pow(2)) * torch.sum(
            invSigmaYdY.pow(2)))
        #print('Finished backward')

        return gradMu, gradSigmaU, gradBeta, None, None, None,None


bayespos = BayesPosFunction.apply


def loss_function(muTheta, logvarTheta, x, mu, logvar, annealParam, args):


    (T, B, N) = logvarTheta.size()
    B=args.batchsize

    diffSq = (x - muTheta).pow(2)
    precis = torch.exp(-logvarTheta)

    BCE = 0.5 * torch.sum(logvarTheta + torch.mul(diffSq, precis))
    BCE /= (T*N*B)

    KLD = -0.5 * annealParam * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalise by same number of elements as in reconstruction
    KLD /= (T * N * B)

    return BCE + args.beta*KLD

def loss_function_deterministic(muTheta, x, args):


    (T, _, N) = muTheta.size()
    B=args.batchsize

    diffSq = (x - muTheta).pow(2)


    BCE = 0.5 * torch.sum(diffSq)
    BCE /= (T*N*B)


    return BCE

def loss_function_reconstruction(muTheta, logvarTheta, x):


    (T, B, N) = logvarTheta.size()

    diffSq = (x - muTheta).pow(2)
    precis = torch.exp(-logvarTheta)

    BCE = 0.5 * torch.sum(logvarTheta + torch.mul(diffSq, precis))
    BCE /= (T*N*B)


    return BCE

def multitask_loss_function(muTheta, logvarTheta, x, excTrue, excPredict, scarTrue, scarPredict, mu, logvar, annealParam):
    (T, B, N) = logvarTheta.size()

    diffSq = (x - muTheta).pow(2)
    precis = torch.exp(-logvarTheta)

    logLikeli = 0.5 * torch.sum(logvarTheta + torch.mul(diffSq, precis))
    logLikeli /= (T * N * B)

    KLD = -0.5 * annealParam * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalise by same number of elements as in reconstruction
    KLD /= (T * N * B)

    BCE_exc=F.binary_cross_entropy(excPredict,excTrue)
    BCE_scar=F.binary_cross_entropy(scarPredict, scarTrue)
    return logLikeli + KLD +BCE_exc + BCE_scar


def rvs(dim=3):
    random_state = np.random
    H = np.eye(dim)
    D = np.ones((dim,))
    for n in range(1, dim):
        x = random_state.normal(size=(dim - n + 1,))
        D[n - 1] = np.sign(x[0])
        x[0] -= D[n - 1] * np.sqrt((x * x).sum())
        # Householder transformation
        Hx = (np.eye(dim - n + 1) - 2. * np.outer(x, x) / (x * x).sum())
        mat = np.eye(dim)
        mat[n - 1:, n - 1:] = Hx
        H = np.dot(H, mat)
        # Fix the last sign such that the determinant is 1
    D[-1] = (-1) ** (1 - (dim % 2)) * D.prod()
    # Equivalent to np.dot(np.diag(D), H) but faster, apparently
    H = (D * H.T).T
    return H
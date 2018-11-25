from __future__ import print_function
# import argparse
import torch
import torch.utils.data
import os
import scipy.io as sio
from torch import nn, optim
from torch.autograd import Variable
from torch.autograd import Function
from torch.autograd.function import once_differentiable

class sssVAE(nn.Module):
    def __init__(self, input_dim, output_dim, mid_dim_i=60, mid_dim_o=800, latent_dim=20):
        super(sssVAE, self).__init__()

        self.fc1 = nn.LSTM(input_dim, mid_dim_i)
        self.fc21 = nn.LSTM(mid_dim_i, latent_dim)
        self.fc22 = nn.LSTM(mid_dim_i, latent_dim)
        self.fc3 = nn.LSTM(latent_dim, mid_dim_o)
        self.fc41 = nn.LSTM(mid_dim_o, output_dim)
        self.fc42 = nn.LSTM(mid_dim_o, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def encode(self, x):
        out, hidden=self.fc1(x)
        h1 = self.relu(out)
        out21,hidden21=self.fc21(h1)
        out22, hidden22 = self.fc22(h1)
        return out21, out22

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        out3,hidden3=self.fc3(z)
        h3 = self.relu(out3)
        out1,hidden1=self.fc41(h3)
        out2, hidden2 = self.fc42(h3)
        return (out1), (out2)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        muTheta,logvarTheta=self.decode(z)
        return muTheta,logvarTheta, mu, logvar

class sssEncoder(nn.Module):
    def __init__(self, input_dim, mid_dim_i=60, latent_dim=20):
        super(sssEncoder, self).__init__()

        self.fc1 = nn.LSTM(input_dim, mid_dim_i)
        self.fc21 = nn.LSTM(mid_dim_i, latent_dim)
        self.fc22 = nn.LSTM(mid_dim_i, latent_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)


    def forward(self, x):
        out, hidden=self.fc1(x)
        h1 = self.relu(out)
        out21,hidden21=self.fc21(h1)
        out22, hidden22 = self.fc22(h1)
        return out21, out22

class svsVAE(nn.Module):
    def __init__(self, input_dim, output_dim, seq_length, mid_dim_i=60, mid_dim_o=800, latent_dim=20, v_mid=800, v_latent=40):
        super(svsVAE, self).__init__()

        self.fc1 = nn.LSTM(input_dim, mid_dim_i)
        self.fc21 = nn.LSTM(mid_dim_i, latent_dim)
        self.fc22 = nn.LSTM(mid_dim_i, latent_dim)
        self.fc3 = nn.LSTM(latent_dim, mid_dim_o)
        self.fc41 = nn.LSTM(mid_dim_o, output_dim)
        self.fc42 = nn.LSTM(mid_dim_o, output_dim)
        self.relu = nn.ReLU()
        self.lin1 =nn.Linear(latent_dim*seq_length, v_mid)
        self.lin2 =nn.Linear(v_mid,v_latent)
        self.lin3 = nn.Linear(v_latent,v_mid)
        self.lin4 = nn.Linear(v_mid, latent_dim*seq_length)
        self.sigmoid = nn.Sigmoid()
        self.latent_dim=latent_dim


    def encode(self, x):
        (_,B,_)=x.size()
        out, hidden=self.fc1(x)
        h1 = self.relu(out)
        out21,hidden21=self.fc21(h1)
        outMean=out21.permute(1,2,0).contiguous().view(B,-1)
        outMean=self.relu(self.lin1(outMean))
        outMean=self.relu(self.lin2(outMean))
        out22, hidden22 = self.fc22(h1)
        outVar = out22.permute(1, 2, 0).contiguous().view(B, -1)
        outVar = self.relu(self.lin1(outVar))
        outVar = self.relu(self.lin2(outVar))
        return outMean, outVar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        (B,_)=z.size()
        z1=self.relu(self.lin3(z))
        z2= self.relu(self.lin4(z1))
        z=z2.view(B,self.latent_dim,-1).permute(2,0,1)

        out3,hidden3=self.fc3(z)
        h3 = self.relu(out3)
        out1,hidden1=self.fc41(h3)
        out2, hidden2 = self.fc42(h3)
        return (out1), (out2)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        muTheta,logvarTheta=self.decode(z)
        return muTheta,logvarTheta, mu, logvar


class svsEncoder(nn.Module):
    def __init__(self, input_dim, seq_length, mid_dim_i=60, latent_dim=20, v_mid=800, v_latent=40):
        super(svsEncoder, self).__init__()

        self.fc1 = nn.LSTM(input_dim, mid_dim_i)
        self.fc21 = nn.LSTM(mid_dim_i, latent_dim)
        self.fc22 = nn.LSTM(mid_dim_i, latent_dim)


        self.relu = nn.ReLU()
        self.lin1 =nn.Linear(latent_dim*seq_length, v_mid)
        self.lin2 =nn.Linear(v_mid,v_latent)

        self.sigmoid = nn.Sigmoid()


    def reparameterize(self, mu, logvar):

        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)

    def forward(self, x):
        (_,B,_)=x.size()
        out, hidden=self.fc1(x)
        h1 = self.relu(out)
        out21,hidden21=self.fc21(h1)
        outMean=out21.permute(1,2,0).contiguous().view(B,-1)
        outMean=self.relu(self.lin1(outMean))
        outMean=self.relu(self.lin2(outMean))
        out22, hidden22 = self.fc22(h1)
        outVar = out22.permute(1, 2, 0).contiguous().view(B, -1)
        outVar = self.relu(self.lin1(outVar))
        outVar = self.relu(self.lin2(outVar))
        return outMean, outVar


class svsVAE_deterministic(nn.Module):
    def __init__(self, input_dim, output_dim, seq_length, mid_dim_i=60, mid_dim_o=800, latent_dim=20, v_mid=800, v_latent=40):
        super(svsVAE_deterministic, self).__init__()

        self.fc1 = nn.LSTM(input_dim, mid_dim_i)
        self.fc21 = nn.LSTM(mid_dim_i, latent_dim)

        self.fc3 = nn.LSTM(latent_dim, mid_dim_o)
        self.fc41 = nn.LSTM(mid_dim_o, output_dim)

        self.relu = nn.ReLU()
        self.lin1 =nn.Linear(latent_dim*seq_length, v_mid)
        self.lin2 =nn.Linear(v_mid,v_latent)
        self.lin3 = nn.Linear(v_latent,v_mid)
        self.lin4 = nn.Linear(v_mid, latent_dim*seq_length)
        self.sigmoid = nn.Sigmoid()
        self.latent_dim=latent_dim


    def encode(self, x):
        (_,B,_)=x.size()
        out, hidden=self.fc1(x)
        h1 = self.relu(out)
        out21,hidden21=self.fc21(h1)
        outMean=out21.permute(1,2,0).contiguous().view(B,-1)
        outMean=self.relu(self.lin1(outMean))
        outMean=self.relu(self.lin2(outMean))

        return self.sigmoid(outMean)



    def decode(self, z):
        (B,_)=z.size()
        z1=self.relu(self.lin3(z))
        z2= self.relu(self.lin4(z1))
        z=z2.view(B,self.latent_dim,-1).permute(2,0,1)

        out3,hidden3=self.fc3(z)
        h3 = self.relu(out3)
        out1,hidden1=self.fc41(h3)

        return (out1)

    def forward(self, x):
        mu = self.encode(x)
        muTheta=self.decode(mu)
        return muTheta,mu


class svsLanguage_deterministic(nn.Module):
    def __init__(self, input_dim, output_dim, seq_length, mid_dim_i=60, mid_dim_o=800, latent_dim=20, v_mid=800, v_latent=50):
        super(svsLanguage_deterministic, self).__init__()

        self.fc1 = nn.LSTM(input_dim, mid_dim_i)
        self.fc21 = nn.LSTM(mid_dim_i, latent_dim)

        self.lstm3 = nn.LSTMCell(latent_dim, v_latent)
        self.lstm4 = nn.LSTMCell(v_latent, mid_dim_o)
        self.lstm5 = nn.LSTMCell(mid_dim_o, output_dim)

        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(latent_dim * seq_length, v_mid)
        self.lin2 = nn.Linear(v_mid, v_latent)
        self.lin3 = nn.Linear(v_latent, v_mid)
        self.lin4 = nn.Linear(v_mid, latent_dim * seq_length)
        self.sigmoid = nn.Sigmoid()

        self.latent_dim = latent_dim
        self.v_latent=v_latent
        self.seq_length=seq_length
        self.mid_dim_o=mid_dim_o
        self.output_dim=output_dim

        self.idlinear=nn.Linear(v_latent,v_latent)
        self.lineardown1=nn.Linear(v_latent,latent_dim)
        self.lineardown2=nn.Linear(output_dim,latent_dim)


    def encode(self, x):
        (_,B,_)=x.size()
        out, hidden=self.fc1(x)
        h1 = self.relu(out)
        out21,hidden21=self.fc21(h1)
        outMean=out21.permute(1,2,0).contiguous().view(B,-1)
        outMean=self.relu(self.lin1(outMean))
        outMean=self.relu(self.lin2(outMean))

        return outMean

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        outData=[]
        h_t = z
        c_t = self.idlinear(z)
        output=self.lineardown1(z)

        h_t2 = Variable(torch.zeros(z.size(0), self.mid_dim_o))
        c_t2 = Variable(torch.zeros(z.size(0), self.mid_dim_o))
        h_t3 = Variable(torch.zeros(z.size(0), self.output_dim))
        c_t3 = Variable(torch.zeros(z.size(0), self.output_dim))
        if z.is_cuda:
            h_t2=h_t2.cuda()
            c_t2=c_t2.cuda()
            h_t3=h_t3.cuda()
            c_t3=c_t3.cuda()

        for i in range(self.seq_length):  # if we should predict the future
            h_t, c_t = self.lstm3(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm4(h_t, (h_t2, c_t2))
            h_t3, c_t3 = self.lstm5(h_t2, (h_t3, c_t3))
            output = self.lineardown2(h_t3)
            #if 'outData' in locals():
            #    outData = torch.cat((outData, h_t3), 2)
            #else:
            #    outData = h_t3
            outData+=[h_t3]
        outData = torch.stack(outData, 0)
        return outData

    def forward(self, x):
        mu = self.encode(x)
        #z = self.reparameterize(mu, logvar)
        muTheta=self.decode(mu)
        return muTheta, mu

class svsLanguage(nn.Module):
    def __init__(self, input_dim, output_dim, seq_length, mid_dim_i=60, mid_dim_o=800, latent_dim=20, v_mid=800, v_latent=50):
        super(svsLanguage, self).__init__()

        self.fc1 = nn.LSTM(input_dim, mid_dim_i)
        self.fc21 = nn.LSTM(mid_dim_i, latent_dim)
        self.fc22 = nn.LSTM(mid_dim_i, latent_dim)

        self.lstm3 = nn.LSTMCell(latent_dim, v_latent)
        self.lstm4 = nn.LSTMCell(v_latent, mid_dim_o)
        self.lstm5 = nn.LSTMCell(mid_dim_o, output_dim)
        self.lstm6 = nn.LSTMCell(mid_dim_o, output_dim)

        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(latent_dim * seq_length, v_mid)
        self.lin2 = nn.Linear(v_mid, v_latent)
        self.lin3 = nn.Linear(v_latent, v_mid)
        self.lin4 = nn.Linear(v_mid, latent_dim * seq_length)
        self.sigmoid = nn.Sigmoid()

        self.latent_dim = latent_dim
        self.v_latent=v_latent
        self.seq_length=seq_length
        self.mid_dim_o=mid_dim_o
        self.output_dim=output_dim

        self.idlinear=nn.Linear(v_latent,v_latent)
        self.lineardown1=nn.Linear(v_latent,latent_dim)
        self.lineardown2=nn.Linear(output_dim,latent_dim)


    def encode(self, x):
        (_,B,_)=x.size()
        out, hidden=self.fc1(x)
        h1 = self.relu(out)
        out21,hidden21=self.fc21(h1)
        outMean=out21.permute(1,2,0).contiguous().view(B,-1)
        outMean=self.relu(self.lin1(outMean))
        outMean=self.relu(self.lin2(outMean))

        out22, hidden22 = self.fc22(h1)
        outvar = out22.permute(1, 2, 0).contiguous().view(B, -1)
        outvar = self.relu(self.lin1(outvar))
        outlogvar = self.relu(self.lin2(outvar))

        return outMean, outlogvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        outMean=[]
        outVar=[]
        h_t = z
        c_t = self.idlinear(z)
        output=self.lineardown1(z)

        h_t2 = Variable(torch.zeros(z.size(0), self.mid_dim_o))
        c_t2 = Variable(torch.zeros(z.size(0), self.mid_dim_o))
        h_t3=h_tvar = Variable(torch.zeros(z.size(0), self.output_dim))
        c_t3=c_tvar = Variable(torch.zeros(z.size(0), self.output_dim))
        if z.is_cuda:
            h_t2=h_t2.cuda()
            c_t2=c_t2.cuda()
            h_t3=h_t3.cuda()
            c_t3=c_t3.cuda()
            h_tvar=h_tvar.cuda()
            c_tvar=c_tvar.cuda()

        for i in range(self.seq_length):  # if we should predict the future
            h_t, c_t = self.lstm3(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm4(h_t, (h_t2, c_t2))
            h_t3, c_t3 = self.lstm5(h_t2, (h_t3, c_t3))
            h_tvar, c_tvar = self.lstm6(h_t2, (h_tvar, c_tvar))
            output = self.lineardown2(h_t3)
            #if 'outData' in locals():
            #    outData = torch.cat((outData, h_t3), 2)
            #else:
            #    outData = h_t3
            outMean+=[h_t3]
            outVar+=[h_tvar]
        outMean = torch.stack(outMean, 0)
        outVar=torch.stack(outVar,0)
        return outMean, outVar

    def forward(self, x):
        mu,logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        muTheta, logvarTheta=self.decode(z)
        return muTheta,logvarTheta, mu, logvar


class svsLanguage_classic(nn.Module):
    def __init__(self, input_dim, output_dim, seq_length, mid_dim_i=60, mid_dim_o=800, latent_dim=20):
        super(svsLanguage_classic, self).__init__()

        self.fc1 = nn.LSTM(input_dim, mid_dim_i)
        self.fc21 = nn.LSTM(mid_dim_i, latent_dim)
        self.fc22 = nn.LSTM(mid_dim_i, latent_dim)

        self.lstm3 = nn.LSTMCell(latent_dim, latent_dim)
        self.lstm4 = nn.LSTMCell(latent_dim, mid_dim_o)
        self.lstm5 = nn.LSTMCell(mid_dim_o, output_dim)
        self.lstm6 = nn.LSTMCell(mid_dim_o, output_dim)

        self.relu = nn.ReLU()

        self.sigmoid = nn.Sigmoid()

        self.latent_dim = latent_dim

        self.seq_length=seq_length
        self.mid_dim_o=mid_dim_o
        self.output_dim=output_dim

        self.idlinear=nn.Linear(latent_dim,latent_dim)
        #self.lineardown1=nn.Linear(v_latent,latent_dim)
        self.lineardown2=nn.Linear(output_dim,latent_dim)


    def encode(self, x):
        (_,B,_)=x.size()
        out, hidden=self.fc1(x)
        h1 = self.relu(out)
        out_mean,(hidden_mean, cell_mean) =self.fc21(h1)

        out_logvar, (hidden_logvar,cell_logvar)=self.fc22(h1)


        return torch.squeeze(hidden_mean),torch.squeeze(cell_mean), torch.squeeze(hidden_logvar),torch.squeeze(cell_logvar)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z, c_t):
        outMean=[]
        outVar=[]
        h_t = z
        #c_t = self.idlinear(z)
        output=self.idlinear(z)

        h_t2 = Variable(torch.zeros(z.size(0), self.mid_dim_o))
        c_t2 = Variable(torch.zeros(z.size(0), self.mid_dim_o))
        h_t3=h_tvar = Variable(torch.zeros(z.size(0), self.output_dim))
        c_t3=c_tvar = Variable(torch.zeros(z.size(0), self.output_dim))
        if z.is_cuda:
            h_t2=h_t2.cuda()
            c_t2=c_t2.cuda()
            h_t3=h_t3.cuda()
            c_t3=c_t3.cuda()
            h_tvar=h_tvar.cuda()
            c_tvar=c_tvar.cuda()

        for i in range(self.seq_length):
            h_t, c_t = self.lstm3(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm4(h_t, (h_t2, c_t2))
            h_t3, c_t3 = self.lstm5(h_t2, (h_t3, c_t3))
            h_tvar, c_tvar = self.lstm6(h_t2, (h_tvar, c_tvar))
            output = self.lineardown2(h_t3)
            #if 'outData' in locals():
            #    outData = torch.cat((outData, h_t3), 2)
            #else:
            #    outData = h_t3
            outMean+=[h_t3]
            outVar+=[h_tvar]
        outMean = torch.stack(outMean, 0)
        outVar=torch.stack(outVar,0)
        return outMean, outVar

    def forward(self, x):
        mu,mu_c, logvar,logvar_c = self.encode(x)
        z = self.reparameterize(mu, logvar)
        muTheta, logvarTheta=self.decode(z,mu_c)
        return muTheta,logvarTheta, mu, logvar

class svsLanguage_classic_deterministic(nn.Module):
    def __init__(self, input_dim, output_dim, seq_length, mid_dim_i=60, mid_dim_o=800, latent_dim=20):
        super(svsLanguage_classic_deterministic, self).__init__()

        self.fc1 = nn.LSTM(input_dim, mid_dim_i)
        self.fc21 = nn.LSTM(mid_dim_i, latent_dim)
        self.fc22 = nn.LSTM(mid_dim_i, latent_dim)

        self.lstm3 = nn.LSTMCell(latent_dim, latent_dim)
        self.lstm4 = nn.LSTMCell(latent_dim, mid_dim_o)
        self.lstm5 = nn.LSTMCell(mid_dim_o, output_dim)
        #self.lstm6 = nn.LSTMCell(mid_dim_o, output_dim)

        self.relu = nn.ReLU()

        self.sigmoid = nn.Sigmoid()

        self.latent_dim = latent_dim

        self.seq_length=seq_length
        self.mid_dim_o=mid_dim_o
        self.output_dim=output_dim

        self.idlinear=nn.Linear(latent_dim,latent_dim)
        #self.lineardown1=nn.Linear(v_latent,latent_dim)
        self.lineardown2=nn.Linear(output_dim,latent_dim)


    def encode(self, x):
        (_,B,_)=x.size()
        out, hidden=self.fc1(x)
        h1 = self.relu(out)
        out_mean,(hidden_mean, cell_mean) =self.fc21(h1)


        return torch.squeeze(hidden_mean),torch.squeeze(cell_mean)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z, c_t):
        outMean=[]
        outVar=[]
        h_t = z
        #c_t = self.idlinear(z)
        output=self.idlinear(z)

        h_t2 = Variable(torch.zeros(z.size(0), self.mid_dim_o))
        c_t2 = Variable(torch.zeros(z.size(0), self.mid_dim_o))
        h_t3 = Variable(torch.zeros(z.size(0), self.output_dim))
        c_t3 = Variable(torch.zeros(z.size(0), self.output_dim))
        if z.is_cuda:
            h_t2=h_t2.cuda()
            c_t2=c_t2.cuda()
            h_t3=h_t3.cuda()
            c_t3=c_t3.cuda()
            #h_tvar=h_tvar.cuda()
            #c_tvar=c_tvar.cuda()

        for i in range(self.seq_length):
            h_t, c_t = self.lstm3(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm4(h_t, (h_t2, c_t2))
            h_t3, c_t3 = self.lstm5(h_t2, (h_t3, c_t3))
            #h_tvar, c_tvar = self.lstm6(h_t2, (h_tvar, c_tvar))
            output = self.lineardown2(h_t3)

            outMean+=[h_t3]
            #outVar+=[h_tvar]
        outMean = torch.stack(outMean, 0)

        return outMean

    def forward(self, x):
        mu,mu_c= self.encode(x)
        muTheta=self.decode(mu,mu_c)
        return muTheta, mu


class inflater(nn.Module):
    def __init__(self, input_dim, output_dim, mid_dim1=400, mid_dim2=1200):
        super(inflater, self).__init__()
        self.relu=nn.ReLU()
        self.layer1=nn.Linear(input_dim,mid_dim1)
        self.layer2=nn.Linear(mid_dim1,mid_dim2)
        self.layer3=nn.Linear(mid_dim2,output_dim)
        self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        y=self.sigmoid(self.layer3(self.relu(self.layer2(self.relu(self.layer1(x))))))
        return y

class linearDiscriminator(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
        super(linearDiscriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size_1)
        self.map2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.map3 = nn.Linear(hidden_size_2, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.map1(x))
        x = self.relu(self.map2(x))
        #return F.sigmoid(self.map3(x))
        return (self.map3(x))
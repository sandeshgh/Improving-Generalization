import torch
import matplotlib.pyplot as plt
import numpy as np

testErr=torch.load('Output/testArr_ECstochastic_svs_y')
testError=testErr['testError']
trainError=testErr['trainError']

plt.subplot(211)


plt.plot(trainError)

plt.subplot(212)

plt.plot(testError)

plt.show()



np.min(testError)




ind=np.argmin(testError)

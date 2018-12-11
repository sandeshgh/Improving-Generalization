import numpy as np

import matplotlib.pyplot as plt


Mean_matrix=np.load('Mean_metric_all_15_25.npy')
Std_matrix= np.load('Std_metric_all_15_25.npy')

fig=plt.figure(1)

xaxis=np.array(range(1,np.size(Mean_matrix,1)+1))
print(xaxis)
print(Mean_matrix[0,:])
print(Std_matrix[0,:])
ax1=fig.add_subplot(221)
ax1.errorbar(xaxis, Mean_matrix[0,:], yerr = Std_matrix[0,:], color="b", label='Stochastic')
ax1.errorbar(xaxis, Mean_matrix[1,:], yerr = Std_matrix[1,:], color="r", label='Deterministic')

ax2=fig.add_subplot(222)
ax2.errorbar(xaxis, Mean_matrix[2,:], yerr = Std_matrix[2,:], color="b", label='Stochastic')
ax2.errorbar(xaxis, Mean_matrix[3,:], yerr = Std_matrix[3,:], color="r", label='Deterministic')

ax3=fig.add_subplot(223)
ax3.errorbar(xaxis, Mean_matrix[4,:], yerr = Std_matrix[4,:], color="b", label='Stochastic')
ax3.errorbar(xaxis, Mean_matrix[5,:], yerr = Std_matrix[5,:], color="r", label='Deterministic')

ax4=fig.add_subplot(224)
ax4.errorbar(xaxis, Mean_matrix[6,:], yerr = Std_matrix[6,:], color="b", label='Stochastic')
ax4.errorbar(xaxis, Mean_matrix[7,:], yerr = Std_matrix[7,:], color="r", label='Deterministic')

#plt.savefig('Fig_18_22')
plt.show()
plt.close()

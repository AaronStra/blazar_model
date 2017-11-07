from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.ticker as ticker
import astropy.units as u
from ssc_model import model, numerics, constants
import naima


energy=np.logspace(10,14,400)*u.eV
frequency = energy*(1.6022e-19)/(6.6261e-34)
z_list=[0.05,0.055]
trans_list=[]

for z in z_list:
    ebl_model = naima.models.EblAbsorptionModel(z, ebl_absorption_model='Dominguez')
    taus=ebl_model(energy)
    trans_list.append(taus)
tau_array=np.array(trans_list)
tau_rel=[]
for i in [0]:
    tau_rel.append(abs(np.expm1(-(tau_array[i+1]-tau_array[i]))))

print(tau_rel)



params = {'legend.fontsize': 'x-large',
          'figure.figsize': (10, 10),
          'figure.subplot.bottom': 0.1,
          'figure.subplot.top':0.96,
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
pylab.rcParams.update(params)

fig, axes = plt.subplots(2, 1)
fig.subplots_adjust(hspace = 0.4)
#font = {'family': 'serif',  'color':'black', 'weight': 'normal', 'size': 16.} # font definitions
for i in [0]:
    axes[1].plot(energy, tau_rel[i], label=r'$\frac{T(z=%0.2f)-T(z=%0.3f)}{T(z=%0.2f)}$'%(z_list[i],z_list[i+1],z_list[i]))
    axes[1].tick_params(which='both', right=True)
for i in range(len(z_list)):
    axes[0].plot(energy, np.exp(-tau_array[i]), label='%0.3f' % z_list[i])
#axes.set_xlim(2,5.1)

axes[0].set_xscale('log')
axes[0].set_yscale('log')#axes.set_ylim(-8,-4)
axes[0].set_xlim(1e10,1e14)
axes[0].set_ylim(1e-19,1e1)
axes[0].set_xlabel(r'$E$ $[eV]$')
axes[0].set_ylabel(r'$T$')
axes[0].legend(loc=0, numpoints=1.)


axes[1].set_xscale('log')
axes[1].set_yscale('log')#axes.set_ylim(-8,-4)
axes[1].set_xlim(1e10,1e14)
axes[1].set_ylim(1e-6,1e1)
axes[1].set_xlabel(r'$E$ $[eV]$')
axes[1].set_ylabel(r'$\sigma_{rel,T}$')
axes[1].legend(loc=0, numpoints=1.)

plt.show()






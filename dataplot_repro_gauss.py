from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.ticker as ticker
import astropy.units as u
from ssc_model import model, numerics, constants

Data_Particles=np.load('./Results/Results_Particles_gauss_002t20b.npy').item()
Data_Particles2=np.load('./Results/Results_Particles_gauss_005t50b.npy').item()
Data_Particles3=np.load('./Results/Results_Particles_gauss_007t70b.npy').item()
Data_Particles4=np.load('./Results/Results_Particles_gauss_02t50b.npy').item()
Data_Particles5=np.load('./Results/Results_Particles_gauss_05t500b.npy').item()
Data_Particles6=np.load('./Results/Results_Particles_gauss_2t200b.npy').item()
Data_Particles7=np.load('./Results/Results_Particles_gauss_4t400b.npy').item()
Data_Particles8=np.load('./Results/Results_Particles_gauss_6t600b.npy').item()
Data_Particles9=np.load('./Results/Results_Particles_gauss_8t800b.npy').item()
Data_Particles10=np.load('./Results/Results_Particles_gauss_10t1000b.npy').item()
Data_Radiation=np.load('./Results/Results_Radiation_gauss_testx.npy').item()

N_e=Data_Particles['N_e_grid']
N_e2=Data_Particles2['N_e_grid']
N_e3=Data_Particles3['N_e_grid']
N_e4=Data_Particles4['N_e_grid']
N_e5=Data_Particles5['N_e_grid']
N_e6=Data_Particles6['N_e_grid']
N_e7=Data_Particles7['N_e_grid']
N_e8=Data_Particles8['N_e_grid']
N_e9=Data_Particles9['N_e_grid']
N_e10=Data_Particles10['N_e_grid']

N_e_list=[N_e[-1],N_e2[-1] ,N_e3[-1],N_e4[-1],N_e5[-1],N_e6[-1],N_e7[-1],N_e8[-1],N_e9[-1],N_e10[-1]]
gamma_grid=Data_Particles['gamma_grid']
print(gamma_grid)
energy=Data_Radiation['energy']
frequency = energy*(1.6022e-19)/(6.6261e-34)

gamma_grid_dict=Data_Particles['gamma_grid_dict']
time_grid = Data_Particles['time_grid']
emission_region = Data_Particles['emission_region']
injected_spectrum = Data_Particles['injected_spectrum']
SSC = model(time_grid, gamma_grid_dict, emission_region, injected_spectrum)
dist = 0#SSC.distance
num = numerics(SSC)

#k_table=num.absorption(N_e, U_rad=False)
#k=k_table(energy)
absorption_factor = SSC.R#num.absorption_factor(k)

volume=4 / 3 * np.pi * SSC.R ** 3
transversion = 1/(4*np.pi*volume*energy.to('erg'))


params = {'legend.fontsize': 'x-large',
          'figure.figsize': (10, 10),
          'figure.subplot.bottom': 0.1,
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
pylab.rcParams.update(params)






fig, axes = plt.subplots(1, 1)
fig.subplots_adjust(hspace = 0.4)
#font = {'family': 'serif',  'color':'black', 'weight': 'normal', 'size': 16.} # font definitions
label_list=[0.02,0.05,0.07,0.2,0.5,2,4,6,8,10]
for i in range(10):
    axes.plot(np.log10(gamma_grid), np.log10(N_e_list[i]), label=label_list[i])

axes.set_xlim(2,5.1)
axes.set_ylim(-8,-4)
axes.set_xlabel(r'$\log(\gamma)$')
axes.set_ylabel(r'$\log(N_e)$')
axes.legend(loc=0, numpoints=1.)
axes.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
axes.xaxis.set_major_locator(ticker.MultipleLocator(1))
axes.yaxis.set_minor_locator(ticker.MultipleLocator(0.2))
axes.yaxis.set_major_locator(ticker.MultipleLocator(1))
axes.tick_params(which='both',top=True, right=True, direction='in', pad=10)
axes.tick_params(which='minor',length=7.5,width=0.9,top=True, right=True, direction='in', pad=10)
axes.tick_params(which='major',length=15, width=1,top=True, right=True, direction='in', pad=10)



plt.show()



params = {'legend.fontsize': 'large',
          'figure.figsize': (10, 10),
          'figure.subplot.bottom': 0.1,
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
pylab.rcParams.update(params)

fig, axes = plt.subplots(2, 1)
fig.subplots_adjust(hspace = 0.4)
#font = {'family': 'serif',  'color':'black', 'weight': 'normal', 'size': 16.} # font definitions
label_list=[0.02,0.05,0.07,0.2,0.5,2,4,6,8,10]
#for i in [20,50,70,200,500,2000,4000,6000,8000,10000]:#20,50,70,200]:#,19,49,199]:#range(0,len(N_e),10):
color_map=['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9']
for i in range(10):
    Syn = num.synchrotron(N_e_list[i])
    IC = num.inverse_compton(N_e_list[i], Syn, absorption_factor)

    SED_SYN = absorption_factor * transversion * Syn.sed(energy, distance=0) * constants.h * frequency
    SED_IC = SSC.R * transversion * IC.sed(energy, distance=0) * constants.h * frequency
    SED = SED_SYN + SED_IC

    if i<=2:
        axes[0].plot(np.log10(frequency.value), np.log10(SED.value), label=label_list[i], color = color_map[i])
    else:
        axes[1].plot(np.log10(frequency.value), np.log10(SED.value), label=label_list[i], color = color_map[i])

axes[0].legend(loc=0, numpoints=1.)
axes[0].set_xlabel(r'$\log(\mathrm{\nu})$')
axes[0].set_ylabel(r'$\log(\mathrm{\nu I(\nu)})$')
axes[0].set_ylim(-5, 8)
axes[0].set_xlim(8, 27)
axes[0].xaxis.set_minor_locator(ticker.MultipleLocator(1))
axes[0].xaxis.set_major_locator(ticker.MultipleLocator(5))
axes[0].yaxis.set_minor_locator(ticker.MultipleLocator(1))
axes[0].yaxis.set_major_locator(ticker.MultipleLocator(5))
axes[0].tick_params(which='both',top=True, right=True, direction='in', pad=10)
axes[0].tick_params(which='minor',length=7.5,width=0.9,top=True, right=True, direction='in', pad=10)
axes[0].tick_params(which='major',length=15, width=1,top=True, right=True, direction='in', pad=10)

axes[1].legend(loc=0, numpoints=1.)
axes[1].set_xlabel(r'$\log(\mathrm{\nu})$')
axes[1].set_ylabel(r'$\log(\mathrm{\nu I(\nu)})$')
axes[1].set_ylim(-11, 8)
axes[1].set_xlim(8, 27)
axes[1].xaxis.set_minor_locator(ticker.MultipleLocator(1))
axes[1].xaxis.set_major_locator(ticker.MultipleLocator(5))
axes[1].yaxis.set_minor_locator(ticker.MultipleLocator(1))
axes[1].yaxis.set_major_locator(ticker.MultipleLocator(5))
axes[1].tick_params(which='both',top=True, right=True, direction='in', pad=10)
axes[1].tick_params(which='minor',length=7.5,width=0.9,top=True, right=True, direction='in', pad=10)
axes[1].tick_params(which='major',length=15, width=1,top=True, right=True, direction='in', pad=10)



plt.show()





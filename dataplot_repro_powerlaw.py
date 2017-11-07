from __future__ import division
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.ticker as ticker
import astropy.units as u
from ssc_model import model, numerics, constants
from astropy.cosmology import WMAP9 as cosmo


Data_Particles1=np.load('./Results/Results_Particles_powerlaw_3t300b.npy').item()



Data_Radiation=np.load('./Results/Results_Radiation_powerlaw_3t300b.npy').item()

N_e1=Data_Particles1['N_e_grid']



N_e_list=N_e1#[N_e1[-1],N_e2[-1] ,N_e3[-1],N_e4[-1],N_e5[-1],N_e6[-1],N_e7[-1],N_e8[-1],N_e9[-1]]
cooling=Data_Particles1['cooling']
U_rad=Data_Particles1['U_rad']


gamma_grid=Data_Particles1['gamma_grid']
energy = np.logspace(-13, 15, 200) * u.eV
frequency = energy*(1.6022e-19)/(6.6261e-34)

gamma_grid_dict=Data_Particles1['gamma_grid_dict']
time_grid = Data_Particles1['time_grid']
emission_region = Data_Particles1['emission_region']
emission_region['z']=1
z=emission_region['z']
dist = cosmo.comoving_distance(z)
dist=dist.value
print(dist)
injected_spectrum = Data_Particles1['injected_spectrum']
SSC = model(time_grid, gamma_grid_dict, emission_region, injected_spectrum)

num = numerics(SSC)

#k_table=num.absorption(N_e, U_rad=False)
#k=k_table(energy)
absorption_factor = SSC.R#num.absorption_factor(k)

volume=4 / 3 * np.pi * SSC.R ** 3
transversion = 1/(4*np.pi*volume*energy.to('erg'))


params = {'legend.fontsize': 'x-large',
          'figure.figsize': (10, 5),
          'figure.subplot.bottom': 0.15,
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
pylab.rcParams.update(params)




#cooling[i][:-1]
#N_e_list[i*2+10]
fig, axes = plt.subplots(1, 1)
fig.subplots_adjust(hspace = 0.2)
#font = {'family': 'serif',  'color':'black', 'weight': 'normal', 'size': 16.} # font definitions

for i in [1,150,170,201]:
    U_rad_time_grid=[]
    for j in range(300):
        U_rad_time = U_rad[j][i]
        U_rad_time_grid.append(U_rad_time)

    axes.plot(SSC.time_grid/(SSC.R/constants.c), U_rad_time_grid, label='%0.1e'%SSC.gamma_grid_midpts[i])

#axes[0].set_xlim(0,5)
#axes[0].get_xaxis().set_visible(False)
axes.set_ylabel(r'$U_{ph}$ $[erg$ $cm^{-3}]$')
axes.set_xlabel(r'$t$ $[\frac{R}{c}]$')
axes.legend(loc=0, numpoints=1.)
#axes.tick_params(which='both',top=True, pad=10)
#axes[0].xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
#axes[0].xaxis.set_major_locator(ticker.MultipleLocator(1))
#axes[0].yaxis.set_minor_locator(ticker.MultipleLocator(0.2))
#axes[0].yaxis.set_major_locator(ticker.MultipleLocator(1))
#axes.tick_params(which='both', right=True, direction='out', pad=10)
#axes[0].tick_params(which='minor',length=7.5,width=0.9, right=True, direction='in', pad=10)
#axes[0].tick_params(which='major',length=15, width=1, right=True, direction='in', pad=10)

plt.show()

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (10, 5),
          'figure.subplot.bottom': 0.1,
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)


fig, axes = plt.subplots(1, 1)
fig.subplots_adjust(hspace = 0.05)
rgb=np.array([0.5,0.7,1])
factor = np.linspace(0.1,1,7)
color_list=rgb*np.vstack(factor)
print(color_list[2])
#color_list=[rgb*4,rgb*4.5,rgb*5,rgb*5.5,rgb*6,rgb*6.5,rgb*7]
o = 0
for i in [0,20,50,100,150,200,300]:

    axes.plot(gamma_grid, N_e1[i], label=i/100, color=color_list[-(o+1)])
    o=o+1
#for i in [100,]:
#    axes[1].plot(gamma_grid, N_e1[i], label=i/100)

#for i in [200,300]:
#    axes[2].plot(gamma_grid, N_e1[i], label=i/100)
#axes.set_ylim(-8,-4)

#axes[0].set_xlim(0,5)
#axes[0].get_xaxis().set_visible(False)
axes.set_ylabel(r'$N_e$ $[cm^{-3}]$')
axes.set_xlabel(r'$\gamma$')
#axes.get_xaxis().set_visible(False)
axes.legend(loc=0, numpoints=1.)
axes.set_xscale('log')
axes.set_yscale('log')
#axes[0].tick_params(which='both',top=True, pad=10)
#axes[0].xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
#axes[0].xaxis.set_major_locator(ticker.MultipleLocator(1))
#axes[0].yaxis.set_minor_locator(ticker.MultipleLocator(0.2))
#axes[0].yaxis.set_major_locator(ticker.MultipleLocator(1))
#axes[0].tick_params(which='both', right=True, direction='out', pad=10)
#axes[0].tick_params(which='minor',length=7.5,width=0.9, right=True, direction='in', pad=10)
#axes[0].tick_params(which='major',length=15, width=1, right=True, direction='in', pad=10)

#axes[1].set_xlim(0,5)
#axes[1].set_xscale('log')
#axes[1].set_yscale('log')
#axes[1].get_xaxis().set_visible(False)
#axes[1].set_xlabel(r'$\gamma$')
#axes[1].set_ylabel(r'$U_{ph}$ $[erg$ $cm^{-3}]$')
#axes[1].legend(loc=0, numpoints=1.)
#axes[1].get_yaxis().set_label_coords(-0.2,0.5)
#axes[1].xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
#axes[1].xaxis.set_major_locator(ticker.MultipleLocator(1))
#axes[1].yaxis.set_minor_locator(ticker.MultipleLocator(0.2))
#axes[1].yaxis.set_major_locator(ticker.MultipleLocator(1))
#axes[1].tick_params(which='both', right=True, direction='out', pad=10)
#axes[1].tick_params(which='minor',length=7.5,width=0.9, right=True, direction='in', pad=10)
#axes[1].tick_params(which='major',length=15, width=1, right=True, direction='in', pad=10)

#axes[2].set_xlim(0,5)
#axes[2].set_xlabel(r'$\gamma$')
#axes[2].set_ylabel(r'$U_{ph}$ $[erg$ $cm^{-3}]$')
#axes[2].legend(loc=0, numpoints=1.)
#axes[2].set_xscale('log')
#axes[2].set_yscale('log')
#axes[2].xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
#axes[2].xaxis.set_major_locator(ticker.MultipleLocator(1))
#axes[2].yaxis.set_major_locator(ticker.MultipleLocator(2e-5))
#axes[2].tick_params(which='both', right=True, direction='out', pad=10)
#axes[2].tick_params(which='minor',length=7.5,width=0.9, right=True, direction='in', pad=10)
#axes[2].tick_params(which='major',length=15, width=1, right=True, direction='in', pad=10)

plt.show()



params = {'legend.fontsize': 'x-large',
          'figure.figsize': (10, 5),
          'figure.subplot.bottom': 0.1,
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
pylab.rcParams.update(params)

fig, axes = plt.subplots(1, 1)
fig.subplots_adjust(hspace = 0.4)
#font = {'family': 'serif',  'color':'black', 'weight': 'normal', 'size': 16.} # font definitions

#for i in [20,50,70,200,500,2000,4000,6000,8000,10000]:#20,50,70,200]:#,19,49,199]:#range(0,len(N_e),10):
for i in [100]:
    Syn = num.synchrotron(N_e_list[i])
    IC = num.inverse_compton(N_e_list[i], Syn, absorption_factor)

    k_table=num.absorption(N_e_list[i], U_rad=False)
    k=k_table(energy)
    absorption_factor_Syn = num.absorption_factor(k)
    absorption_factor_IC = SSC.R

    #I_SYN = absorption_factor_Syn * transversion * Syn.sed(energy, distance=0) * constants.h * frequency
    #I_IC = absorption_factor_IC * transversion * IC.sed(energy, distance=0) * constants.h * frequency

    SED_SYN=Syn.sed(energy, distance=dist*u.Mpc)*absorption_factor_Syn/SSC.R
    SED_IC=IC.sed(energy, distance=dist*u.Mpc)*absorption_factor_IC/SSC.R
    for j in range(len(SED_SYN)):
        if SED_SYN[j]==0:
            SED_SYN[j]=1e-100*u.Unit('eV/(cm2 s)')
    SED = SED_SYN + SED_IC

    np.save('./Results/Results_Particles_powerlaw_SED.npy', SED.value)
    #SED = np.load('./Results/Results_Particles_powerlaw_SED.npy')
    boosted_energy, boosted_SED = num.doppler(energy, SED)
    final_SED = num.ebl(boosted_energy, boosted_SED)
    boosted_frequency = boosted_energy*1.62e-19/(6.63e-34)

    axes.plot(energy.value, SED, label='intrinsic')
    axes.plot(boosted_energy.value, boosted_SED, label='doppler boosted', linestyle=':', lw=2)
    axes.plot(boosted_energy.value, final_SED, label="final")
    #else:
    #    axes[1].plot(np.log10(frequency.value), np.log10(SED.value), label=i/100)

axes.legend(loc=0, numpoints=1.)
axes.set_xlabel(r'$E$ $[eV]$')
axes.set_ylabel(r'$E\times\frac{\mathrm{d}F}{\mathrm{d}E}$ $[erg$ $cm^{-2}$ $s^{-1}]$')
#axes.set_ylim(0.5, 7)
axes.set_xlim(1e-13, 1e15)
axes.set_xscale('log')
axes.set_yscale('log')
#axes.xaxis.set_minor_locator(ticker.MultipleLocator(1))
#axes.xaxis.set_major_locator(ticker.MultipleLocator(5))
#axes.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
#axes.yaxis.set_major_locator(ticker.MultipleLocator(2))
#axes.tick_params(which='both',top=True, right=True, direction='in', pad=10)
#axes.tick_params(which='minor',length=7.5,width=0.9,top=True, right=True, direction='in', pad=10)
#axes.tick_params(which='major',length=15, width=1,top=True, right=True, direction='in', pad=10)

#axes[1].legend(loc=0, numpoints=1.)
#axes[1].set_xlabel(r'$\log(\mathrm{\nu})$')
#axes[1].set_ylabel(r'$\log(\mathrm{\nu I(\nu)})$')
#axes[1].set_ylim(0.5, 7)
#axes[1].set_xlim(9, 26)
#axes[1].xaxis.set_minor_locator(ticker.MultipleLocator(1))
#axes[1].xaxis.set_major_locator(ticker.MultipleLocator(5))
#axes[1].yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
#axes[1].yaxis.set_major_locator(ticker.MultipleLocator(2))
#axes[1].tick_params(which='both',top=True, right=True, direction='in', pad=10)
#axes[1].tick_params(which='minor',length=7.5,width=0.9,top=True, right=True, direction='in', pad=10)
#axes[1].tick_params(which='major',length=15, width=1,top=True, right=True, direction='in', pad=10)



plt.show()





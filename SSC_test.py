################################################################################
# reordering and cleaning SSC_test for the upload on github
# testing the SSC_class
# 19 May 2017
# author: Cosimo Nigro (cosimonigro2@gmail.com)
################################################################################

import numpy as np
from math import pi
import matplotlib.pyplot as plt
from ssc_model import model, numerics, constants
import astropy.units as u
import timeit
from astropy.table import Table
import matplotlib.pylab as pylab


file=open('datatofit.dat')
f = Table.read(file.read(), format='ascii')




time_grid = dict(time_min = 0, time_max = 0.02, time_bins = 2)
gamma_grid = dict(gamma_min =1, gamma_max = 1e5, gamma_bins =10)
#gamma_grid = dict(gamma_min =2, gamma_max = 1e5, gamma_bins =200)
emission_region = dict(R = 1e16, B = 1, t_esc = 1.5, gamma = 10, theta = 0, z = 0.047)
injected_spectrum = dict(type = 'power-law', norm = 1e-3, alpha=-1.7, t_inj = 1)
#injected_spectrum = dict(type='power-law', norm=1e3, alpha=-1.7, t_inj=0.1)

start = timeit.default_timer()
# let us initialize the ssc object
SSC = model(time_grid, gamma_grid, emission_region, injected_spectrum)
dist = 0#SSC.distance
num = numerics(SSC)
# let us evolve it
N_e = num.evolve()


#k2=absorption(N_e, U_rad=True)

energy = np.logspace(-13, 15, 200) * u.eV
frequency = energy*(1.6022e-19)/(6.6261e-34)#*5e-5
#k_table=num.absorption(N_e, U_rad=False)
#k=k_table(energy)
absorption_factor = SSC.R#num.absorption_factor(k)
Syn = num.synchrotron(N_e)
IC = num.inverse_compton(N_e,Syn,absorption_factor)
volume=4 / 3 * pi * SSC.R ** 3





transversion = 1/(4*np.pi*volume*energy.to('erg'))
SED_SYN = absorption_factor*transversion*Syn.sed(energy, distance=0)*constants.h*frequency
SED_IC = SSC.R*transversion*IC.sed(energy, distance=0)*constants.h*frequency
SED = SED_SYN+SED_IC
# calculate the SED
# fetch the naima inverse compton and synchrotron object
#SYN = num.synchrotron(N_e, SSC.energy_grid)
#IC = num.inverse_compton(N_e)
#SED = SYN.sed(energy, dist)+IC.sed(energy, dist)
#boosted_energy, boosted_SED = num.doppler(energy, SED)
#final_SED = num.ebl(boosted_energy, boosted_SED)
#boosted_frequency = boosted_energy*1.62e-19/(6.63e-34)



stop = timeit.default_timer()
print 'Computational time: '
print stop - start, ' s'


# Saving Results
Results_Particles = dict(cooling = num.cooling, U_rad=num.U_rad_grid, N_e_grid = num.N_e_grid, time_grid = time_grid, gamma_grid_dict = gamma_grid,
                         gamma_grid= SSC.gamma_grid, emission_region=emission_region, injected_spectrum=injected_spectrum)
Results_Radiation = dict(energy = energy, frequency=frequency, absorption_factor = absorption_factor, SED_Syn = SED_SYN, SED_IC = SED_IC, SED = SED)
np.save('./Results/Results_Particles_powerlaw_3t30b.npy', Results_Particles)
np.save('./Results/Results_Radiation_powerlaw_3t30b.npy', Results_Radiation)





# plotting section

params = {'legend.fontsize': 'large',
          'figure.figsize': (10, 10),
          'figure.subplot.bottom': 0.2,
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
pylab.rcParams.update(params)

fig, axes = plt.subplots(2, 1)
fig.subplots_adjust(hspace = 0.4)
font = {'family': 'serif',  'color':'black', 'weight': 'normal', 'size': 16.} # font definitions




# first plot with electron spectrum
for i in range(len(num.N_e_grid)):
    axes[1].plot(np.log10(SSC.gamma_grid), np.log10(num.N_e_grid[i]),  ls = '-', lw=1, marker = '',
            color = 'black')

#axes[0].legend(loc = 0, numpoints = 1., prop = {'size':12.})
#axes[0].set_xlabel(r'$\gamma$')
#axes[0].set_ylabel(r'$n_{e}$')
axes[1].set_xlim(0, 5)
#axes[1].set_ylim(-8, -4)
#axes[1].set_xscale('log')
#axes[1].set_yscale('log')




# calculate the SED
# fetch the naima inverse compton and synchrotron object



axes[0].plot(np.log10(frequency.value),np.log10(SED_SYN.value))
axes[0].plot(np.log10(frequency.value),np.log10(SED_IC.value))
axes[0].legend(loc = 0, numpoints = 1.)
axes[0].set_xlabel(r'$\log(\mathrm{\nu})$')
axes[0].set_ylabel(r'$\log(\mathrm{\nu I(\nu)})$')
axes[0].set_ylim(1, 7)
axes[0].set_xlim(9, 26)
#axes[0].set_xscale('log')
#axes[0].set_yscale('log')

plt.show()
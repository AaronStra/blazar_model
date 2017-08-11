################################################################################
# reordering and cleaning SSC_test for the upload on github
# testing the SSC_class
# 19 May 2017
# author: Cosimo Nigro (cosimonigro2@gmail.com)
################################################################################

import numpy as np
from math import pi
import matplotlib.pyplot as plt
from ssc_model import model, numerics
import astropy.units as u
import timeit

time_grid = dict(time_min = 0., time_max = 10., time_bins = 200)
gamma_grid = dict(gamma_min = 6e-3, gamma_max = 3.9e6, gamma_bins = 30)
emission_region = dict(R = 5.8e15, B = 0.04, t_esc = 1.5, gamma = 100, theta = 0, z = 1)
injected_spectrum = dict(norm = 2.2e-1, alpha = -2, t_inj = 5.)

start = timeit.default_timer()

# let us initialize the ssc object
SSC = model(time_grid, gamma_grid, emission_region, injected_spectrum)
dist = SSC.distance
num = numerics(SSC)


# let us evolve it
N_e = num.evolve()

# calculate the SED
# fetch the naima inverse compton and synchrotron object
SYN = num.synchrotron(N_e)
IC = num.inverse_compton(N_e)

energy = np.logspace(-7, 13, 200) * u.eV
SED = SYN.sed(energy, dist)+IC.sed(energy, dist)

boosted_energy, boosted_SED = num.doppler(energy, SED)
final_SED = num.ebl(boosted_energy, boosted_SED)

stop = timeit.default_timer()
print 'Computational time: '
print stop - start, ' s'

# plotting section
fig, axes = plt.subplots(3, 1)
fig.subplots_adjust(hspace = 0.4)
font = {'family': 'serif',  'color':'black', 'weight': 'normal', 'size': 16.} # font definitions

# first plot with electron spectrum
axes[0].plot(SSC.gamma_grid, SSC.gamma_grid**2*SSC.N_e_inj,  ls = '--', lw=2, marker = '',
            color = 'turquoise', label = 'Injected Spectrum')
axes[0].plot(SSC.gamma_grid, SSC.gamma_grid**2*N_e,  ls = '-', lw=2, marker = '',
            color = 'crimson', label = 'numerical solution')
axes[0].legend(loc = 0, numpoints = 1., prop = {'size':12.})
axes[0].set_xscale('log')
axes[0].set_xlabel(r'$\gamma$')
axes[0].set_ylabel(r'$\gamma^2 \times n_{e}$')
axes[0].set_yscale('log')




axes[1].plot(energy, SED, lw=3, color='royalblue', label='Synchrotron + Inverse Compton')
axes[1].plot(boosted_energy, boosted_SED, lw=3, color='red', label='boosted SED')
axes[1].plot(boosted_energy, final_SED, lw=3, color='green', label='observers SED')
axes[1].legend(loc = 0, numpoints = 1., prop = {'size':8.})
axes[1].set_xlabel(r'$E\,[eV]$')
axes[1].set_ylabel(r'$E^{2} \times {\rm d}F/{\rm d}E\,[erg\,cm^{-2}\,s^{-1}]$')
axes[1].set_ylim(1e-25, 1e-5)
axes[1].set_xscale('log')
axes[1].set_yscale('log')


axes[2].plot(boosted_energy, final_SED, lw=3, color='green', label='observers SED')
axes[2].legend(loc = 0, numpoints = 1., prop = {'size':8.})
axes[2].set_xlabel(r'$E\,[eV]$')
axes[2].set_ylabel(r'$E^{2} \times {\rm d}F/{\rm d}E\,[erg\,cm^{-2}\,s^{-1}]$')
axes[2].set_ylim(1e-25, 1e-5)
axes[2].set_xscale('log')
axes[2].set_yscale('log')



fig.savefig('SSC_test_output_final.png')
plt.show()

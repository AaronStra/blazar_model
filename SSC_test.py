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

Times = [0.02, 0.05, 0.07, 0.2, 0.5, 2, 4, 6, 8, 10]
i = 0
for time in Times:

    time_grid = dict(time_min = 0., time_max = time, time_bins = 100)
    gamma_grid = dict(gamma_min = 1e-2, gamma_max = 1e6, gamma_bins = 1000)
    emission_region = dict(R = 1e16, B = 1, t_esc = 1.5, gamma = 100, theta = 0, z = 1)
    injected_spectrum = dict(norm = 9e-9, alpha = -2, t_inj = 0.1)

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
    fig, axes = plt.subplots(2, 1)
    fig.subplots_adjust(hspace = 0.4)
    font = {'family': 'serif',  'color':'black', 'weight': 'normal', 'size': 16.} # font definitions

    # first plot with electron spectrum
    axes[0].plot(SSC.gamma_grid, N_e,  ls = '-', lw=1, marker = '',
              color = 'black', label = 'numerical solution')
    axes[0].legend(loc = 0, numpoints = 1., prop = {'size':12.})
    axes[0].set_xscale('log')
    axes[0].set_xlabel(r'$\gamma$')
    axes[0].set_ylabel(r'$n_{e}$')
    axes[0].set_xlim(1e2, 2e5)
    axes[0].set_ylim(1e-8, 1e-4)
    axes[0].set_yscale('log')




    #axes[1].plot(energy, SED, lw=3, color='royalblue', label='Synchrotron + Inverse Compton')
    #axes[1].plot(boosted_energy, boosted_SED, lw=3, color='red', label='boosted SED')
    #axes[1].plot(boosted_energy, final_SED, lw=3, color='green', label='observers SED')
    #axes[1].legend(loc = 0, numpoints = 1., prop = {'size':8.})
    #axes[1].set_xlabel(r'$E\,[eV]$')
    #axes[1].set_ylabel(r'$E^{2} \times {\rm d}F/{\rm d}E\,[erg\,cm^{-2}\,s^{-1}]$')
    #axes[1].set_ylim(1e-25, 1e-5)
    #axes[1].set_xscale('log')
    #axes[1].set_yscale('log')


    axes[1].plot(boosted_energy, final_SED, lw=3, color='green', label='observers SED')
    axes[1].legend(loc = 0, numpoints = 1., prop = {'size':8.})
    axes[1].set_xlabel(r'$E\,[eV]$')
    axes[1].set_ylabel(r'$E^{2} \times {\rm d}F/{\rm d}E\,[erg\,cm^{-2}\,s^{-1}]$')
    #axes[1].set_ylim(1e-200, 1e200)
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_xlim(1e-3, 1e13)
    axes[1].set_ylim(1e-25, 1e-5)


fig.savefig('SSC_test_output_final.png')
plt.show()

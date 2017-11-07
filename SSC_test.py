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

time_grid = dict(time_min = 0, time_max = 1, time_bins =10 )
gamma_grid = dict(gamma_min =1, gamma_max = 1e5, gamma_bins =20)
emission_region = dict(R = 1e16, B = 1, t_esc = 1.5, gamma = 10, theta = 0, z = 0)
injected_spectrum = dict(type = 'power-law', norm = 4e0, alpha=-1.7, t_inj = 1)

start = timeit.default_timer()
# let us initialize the ssc object
SSC = model(time_grid, gamma_grid, emission_region, injected_spectrum)
dist = SSC.distance
num = numerics(SSC)
# let us evolve it
N_e = num.evolve()

# Saving Results
Results_Particles = dict(cooling = num.cooling, U_rad=num.U_rad_grid, N_e_grid = num.N_e_grid, time_grid = time_grid, gamma_grid_dict = gamma_grid,
                         gamma_grid= SSC.gamma_grid, emission_region=emission_region, injected_spectrum=injected_spectrum)
np.save('./Results/Results_Particles_test_x.npy', Results_Particles)


# plotting section, optional
plotting=True

if plotting==False:
    pass
else:
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

    # first plot with final electron spectrum
    axes[0].plot(SSC.gamma_grid, N_e)
    axes[0].legend(loc = 0, numpoints = 1., prop = {'size':12.})
    axes[0].set_xlabel(r'$\gamma$')
    axes[0].set_ylabel(r'$N_{e}$')
    #axes[0].set_xlim(0, 5)
    #axes[0].set_ylim(-8, -4)
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')

    # calculate the final SED
    energy = np.logspace(-13, 15, 200) * u.eV
    frequency = energy * (1.6022e-19) / (6.6261e-34)  # *5e-5
    k_table = num.absorption(N_e, U_rad=False)
    k = k_table(energy)
    absorption_factor = num.absorption_factor(k)
    Syn = num.synchrotron(N_e)
    IC = num.inverse_compton(N_e, Syn, absorption_factor)
    volume = 4 / 3 * pi * SSC.R ** 3

    transversion = 1 / (4 * np.pi * volume * energy.to('erg'))
    SED_SYN = Syn.sed(energy, distance=0)*absorption_factor/SSC.R
    SED_IC = IC.sed(energy, distance=0)
    SED = SED_SYN + SED_IC


    boosted_energy, boosted_SED = num.doppler(energy, SED)
    final_SED = num.ebl(boosted_energy, boosted_SED)

    stop = timeit.default_timer()
    print 'Computational time: '
    print stop - start, ' s'

    axes[1].plot(boosted_energy, final_SED)
    axes[1].legend(loc = 0, numpoints = 1.)
    axes[1].set_xlabel(r'E [eV]')
    axes[1].set_ylabel(r'$E\times\frac{\mathrm{d}F}{\mathrm{d}E}')
    #axes[1].set_ylim(-20, 7)
    #axes[1].set_xlim(-3, 14)
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')

    plt.show()
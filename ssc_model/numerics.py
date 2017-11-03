from __future__ import division
from math import pi, sqrt, exp, log
import numpy as np
from constants import *
import astropy.units as u
import astropy.constants as const
import naima
from naima.utils import trapz_loglog
import matplotlib.pyplot as plt

class numerics:
    '''
    Numeric procedures needed to evaluate SSC emission.
    class to be intialized with a model object (as defined in module.py)
    Reference is Chiaberge & Ghisellini (1998) MNRAS, 306,551 (1999).
    '''


    def __init__(self, model_object):
        # initialize the model object we will use from now on
        self.model = model_object
        self.energy = np.logspace(-13, 15, 200) * u.eV


    def ChaCoo_tridiag_matrix(self,N_e):
        '''
        Implementing tridiagonal matrix of Eq.(9) of Reference
        '''
        # the dimension of our problem is
        N = len(self.model.delta_gamma)
        ChaCoo_matrix = np.zeros((N,N), float)

        # Precalculate quantities for U_rad
        # Calculate particle distribution at gamma_grid_midpts
        N_e_table = naima.models.TableModel(self.model.gamma_grid * u.eV,
                                                   N_e,
                                                   amplitude=1)
        N_e_midpts=N_e_table(self.model.gamma_grid_midpts*u.eV)

        # Setting first and last value equal to the original one
        # to avoid additional zeros at the end and the beginning
        # of the array due to the tablemodelling.
        N_e_midpts[0],N_e_midpts[-1]=N_e[0],N_e[-1]

        # Calculate specific intensity of the radiation based on N_e_midpts
        I_table=self.radiation_field_table(N_e_midpts,U_rad=True,absorption=True)

        # Build U_rad_array
        U_rad = []
        for gamma in self.model.gamma_grid_midpts:
            U_rad_element = self.u_rad_calc(gamma, I_table).value
            U_rad.append(U_rad_element)
        U_rad = np.array(U_rad)

        # Calculate cool_rate for current particle distribution N_e
        cooling=self.cool_rate(self.model.gamma_grid_midpts, U_rad)

        # giving access to the calculated values after the evolution
        self.U_rad_grid.append(U_rad)
        self.cooling.append(cooling)

        # loop on the energy to fill the matrix
        for i in range(N):
            delta_gamma = self.model.delta_gamma[i]
            delta_t = self.model.delta_t
            t_esc = self.model.t_esc

            # Eq.s (10) of Reference
            V2 = 1 + delta_t/t_esc + \
                     (delta_t * cooling[i])/delta_gamma

            V3 = - (delta_t * cooling[i+1])/delta_gamma
            # let's loop on another dimension to fill the diagonal of the matrix
            for j in range(N):
                if j == i:
                    ChaCoo_matrix[i,j] = V2
                if j == i+1:
                    ChaCoo_matrix[i,j] = V3

        # Chang Cooper boundaries condition, Eq.(18) Park Petrosian, 1996
        # http://adsabs.harvard.edu/full/1996ApJS..103..255P
        ChaCoo_matrix[N-2,N-1] = 0.
        return ChaCoo_matrix

    def u_rad_calc(self, gamma, I_table):
        '''
        Calculating U_rad following Eq.(17) of Reference

        Parameters
        ----------
        gamma : float
            Lorentz Factor limiting the Integration Range due to
            Eq(16) of Reference.

        I_table : function
            I_table function, taking photon energies as a
            `~astropy.units.Quantity` array or float, and returning the specific
            intensity in units of 1/(cm2 s) as a
            `~astropy.units.Quantity` array or float.
        '''

        e_max = (3 * m_e * c ** 2 / (4 * gamma) * u.erg).to('eV').value
        energy = np.logspace(-13, np.log10(e_max), 50) * u.eV

        I = I_table(energy)
        U_rad = 4 * np.pi / c * trapz_loglog(I.value, energy.to('erg').value)*u.Unit('erg/cm3')

        return U_rad

    def cool_rate(self,gamma_grid, U_rad):
        '''
        Calculating Cooling Rate from Eq.(2) of Reference

        Parameters
        ----------
        gamma_grid : array
            Array of Lorentz Factors, usually gamma_grid_midpts due to
            the used Chang and Cooper scheme.

        U_rad : array
            Energy densities as an array of float values corresponding
            to the value in units of erg/cm3 having the same length as
            gamma_grid.
        '''

        U_B = np.ones(len(gamma_grid))*self.model.U_B
        return 4 / 3 * sigma_T / (m_e * c) * (U_B + U_rad)* gamma_grid ** 2

    def SingleSynchrotron(self, gamma, B, photon_energy):

        '''Synchrotron emission from a single electron.

        Here the same approximation of synchrotron emissivity in a random magnetic field
        is used as in the naima model for a whole electron population.

        Parameters
        ----------
        gamma : float
            Lorentz factor of the electron corresponding to its energy

        B : :class:`~astropy.units.Quantity` float instance
            Isotropic magnetic field strength.

        photon_energy : :class:`~astropy.units.Quantity` float instance, array
            Photon energy range of relevant emission.
       '''

        outspecene = photon_energy

        from scipy.special import cbrt

        def Gtilde(x):
            """
            AKP10 Eq. D7

            Factor ~2 performance gain in using cbrt(x)**n vs x**(n/3.)
            Invoking crbt only once reduced time by ~40%
            """
            cb = cbrt(x)
            gt1 = 1.808 * cb / np.sqrt(1 + 3.4 * cb ** 2.)
            gt2 = 1 + 2.210 * cb ** 2. + 0.347 * cb ** 4.
            gt3 = 1 + 1.353 * cb ** 2. + 0.217 * cb ** 4.
            return gt1 * (gt2 / gt3) * np.exp(-x)

        CS1_0 = np.sqrt(3) * e ** 3 * B.to('G').value
        CS1_1 = (m_e * c ** 2 * h)
        CS1 = CS1_0 / CS1_1

        # Critical energy, erg
        Ec = 3 * e * h / (2 * pi) * B.to('G').value * gamma ** 2
        Ec /= 2 * (m_e * c)

        EgEc = outspecene.to('erg').value / Ec
        dNdE = CS1 * Gtilde(EgEc)
        # return units
        spec_lum = dNdE * u.Unit('1/(s)')

        return spec_lum

    def absorption(self, N_e, U_rad=False):
        '''
        Calculating absorption coefficient k table model from Eq.(15) of Reference

        Parameters
        ----------
        N_e : array
            Particle distribution as float values corresponding to
            units of 1/cm3. Equals the number of electrons per Lorentz factor.
        U_rad : boolean, optional
            Dependence on the purpose of calculation for the final
            absorption in the SED (False) or the calculation of U_rad (True).
        '''
        if U_rad:
            gamma_grid=self.model.gamma_grid_midpts
            e_max = (3 * m_e * c ** 2 / (4 * gamma_grid[0]) * u.erg).to('eV').value
            energy = np.logspace(-13, np.log10(e_max), 50) * u.eV

        else:
            gamma_grid = self.model.gamma_grid
            energy = self.energy

        C1 = -h ** 3 / (8 * np.pi * m_e)
        k_abs = []

        # Taking care of p > 0.
        if gamma_grid[0] <= 1:
            gamma_grid[0] = 1.02

        p = np.sqrt(gamma_grid ** 2 - 1)

        def funcderiv(energy, gamma):
            '''
            Function inside the derivative depending on the photon energy
            and the Lorentz factor of the electron
            '''
            p = sqrt(gamma ** 2 - 1)
            B = self.model.B*u.G
            Syn = self.SingleSynchrotron(gamma, B, energy)
            return gamma * p * Syn.value

        # Build the absorption coefficient array depending on photon energies
        for eps in energy:
            deriv_list = []
            for gamma in gamma_grid:
                dx = gamma / 1000
                deriv = (-funcderiv(eps, gamma + 2 * dx) + 8 * funcderiv(eps, gamma + dx) - 8
                         * funcderiv(eps,gamma - dx) + funcderiv(eps, gamma - 2 * dx)) / (12 * dx)
                deriv_list.append(deriv)

            deriv_array = np.array(deriv_list)

            int = trapz_loglog(N_e / (p * gamma_grid) * deriv_array, gamma_grid)

            k = C1 / eps.to('erg').value ** 2 * int
            k_abs.append(k)

        k_final = -np.array(k_abs)

        # Use table_model to be flexible in relevant photon energies
        k_final_table = naima.models.TableModel(energy,
                                                k_final,
                                                amplitude=1)
        return k_final_table

    def absorption_factor(self, k):
        '''
        Absorption factor for synchrotron-self absorption as float
        value in units of cm according to Eq.(14).

        Parameters
        ----------
        k : array
            unitless absorption coefficient with value according
            to the unit of 1/cm
        '''

        # avoid division by zero:
        for i in range(len(k)):
            if k[i]<=1e-300:
                k[i]=1e-300
        absorption_factor = 1 / k * (-np.expm1(-k * self.model.R))

        return absorption_factor

    def synchrotron(self, N_e, U_rad = False):
        # we plug now the obtained electron distribution in naima TableModel
        # remultiply by Volume and devide by the electrons rest mass energy to get N_e differential in Energy

        # the used energy_grid depends on if the calculation is for U_rad or the final spectrum
        if U_rad:
            energy_grid = self.model.gamma_grid_midpts*E_rest
        else:
            energy_grid = self.model.energy_grid

        N_e_differential = N_e * self.model.volume / (m_e * c ** 2)
        electron_density = naima.models.TableModel(energy_grid * u.eV,
                                                   N_e_differential * u.Unit('1/erg'),
                                                   amplitude=1)

        SYN = naima.models.Synchrotron(electron_density, B=self.model.B*u.G)

        return SYN


    def inverse_compton(self,N_e, Syn, abs_fac, U_rad=False):

        if U_rad:
            energy_grid = self.model.gamma_grid_midpts*E_rest
        else:
            energy_grid = self.model.energy_grid

        N_e_differential = N_e * self.model.volume/ (m_e * c ** 2)
        electron_density = naima.models.TableModel(energy_grid * u.eV,
                                                    N_e_differential * u.Unit('1/erg'),
                                                    amplitude = 1)

        # Define energy array for synchrotron seed photon field and compute
        # Synchroton luminosity by setting distance to 0.l
        energy = self.energy
        Lsy = Syn.flux(energy, distance=0*u.cm)*abs_fac/self.model.R

        # Define source radius and compute photon density
        R =  self.model.R * u.cm
        phn_sy = Lsy / (4 * np.pi * R**2 * const.c) * 2.26

        # Create IC instance with CMB and synchrotron seed photon fields:
        IC = naima.models.InverseCompton(electron_density, seed_photon_fields=['CMB',['SSC', energy, phn_sy]])

        return IC

    def radiation_field_table(self,N_e, U_rad=False, absorption=True):
        '''
        Calculation of the specific intensity inside the emission region.
        It consists of the Synchrotron and the IC radiation.
        Parameters
        ----------
        N_e : array
            Particle distribution as float values corresponding to
            units of 1/cm3. Equals the number of electrons per Lorentz factor.

        U_rad : boolean, optional
            Dependence on the purpose of calculation for the final
            absorption in the SED (False) or the calculation of U_rad (True).

        absorption : boolean, optional
            Only for test purposes. Turn it off to safe time.
            (For example adding the calculation of the absorption leads to
            the need of four times the computational time of a run without at 200 gamma_bins)
        '''

        transversion=1/(4*np.pi*self.model.volume*self.energy.to('erg'))

        if U_rad:
            if absorption:
                k = self.absorption(N_e, U_rad=True)(self.energy)
                abs_fac=self.absorption_factor(k)
            else:
                abs_fac=self.model.R
            Syn = self.synchrotron(N_e, U_rad=True)
            IC = self.inverse_compton(N_e,Syn, abs_fac,U_rad=True)
        else:
            if absorption:
                k = self.absorption(N_e)(self.energy)
                abs_fac=self.absorption_factor(k)
            else:
                abs_fac=self.model.R
            Syn = self.synchrotron(N_e)
            IC = self.inverse_compton(N_e, Syn, abs_fac)

        eps_syn = Syn.sed(self.energy, distance=0).to('erg/s') * transversion
        eps_ic = IC.sed(self.energy, distance=0).to('erg/s') * transversion

        eps = eps_ic + eps_syn
        I = eps * abs_fac

        I_table = naima.models.TableModel(self.energy,
                                          I.value*u.Unit('1/(cm2 s)'),
                                          amplitude=1)
        return I_table


    def doppler(self, energy, SED):
        '''
        Calculating Doppler boosting based on the intrinsic emission
        and the model parameters.
        '''
        beta = self.model.beta
        theta = self.model.theta

        doppler_factor = np.sqrt(1-beta**2)/(1-beta*np.cos(theta/180*pi))
        boosted_SED = doppler_factor**3*SED
        boosted_energy = doppler_factor*energy


        return boosted_energy, boosted_SED


    def ebl(self, boosted_energy, boosted_SED):
        '''
        Adding absorption due to extra galactic background light with the naima model
        '''
        ebl_model = naima.models.EblAbsorptionModel(self.model.z, ebl_absorption_model='Dominguez')
        transmissivity  = ebl_model.transmission(boosted_energy)
        final_SED = boosted_SED*transmissivity

        return final_SED


    def evolve(self):
        '''
        Evolving injected spectrum solving iteratively Eq.(9).
        We will calculate the synchrotron emissivity with the romberg integration
        and will update the model.U_rad parameter.
        Options only_synchrotron_cooling is for test
        '''

        self.N_e_grid = []
        self.cooling=[]
        self.U_rad_grid=[]

        delta_t = self.model.delta_t

        # injected spectrum
        if self.model.inj_spectr_type=='power-law':
            Q_e = self.model.powerlaw_injection
            N_e = self.model.powerlaw_injection*delta_t
        elif self.model.inj_spectr_type=='broken power-law':
            Q_e = self.model.broken_powerlaw_injection
            N_e = self.model.broken_powerlaw_injection*delta_t
        elif self.model.inj_spectr_type=='gaussian':
            Q_e = self.model.gaussian_injection
            N_e = self.model.gaussian_injection*delta_t

        self.N_e_grid.append(N_e)

        # injecton term, to be added each delta_t up to the maximum injection time
        # specified by model.inj_time



        # time grid loop
        time_past_injection = 0

        for time in self.model.time_grid:

            # here N^{i+1} of Reference --> N_e_tmp
            # here N^{i} of Reference --> N_e
            # solve the system with Eq.(11):
            if time_past_injection <= self.model.inj_time:
                N_e_tmp = np.linalg.solve(self.ChaCoo_tridiag_matrix(N_e), N_e + Q_e*delta_t)
            # no more injection after the established t_inj
            else:
                N_e_tmp = np.linalg.solve(self.ChaCoo_tridiag_matrix(N_e), N_e)

            # swap!, now we are moving to the following istant of time
            # N^{i+1} --> N^{i} and restart
            N_e = N_e_tmp
            self.N_e_grid.append(N_e)
            # update the time past injection
            print 't after injection: ', time_past_injection/self.model.crossing_time, ' crossing time'
            time_past_injection += delta_t

        return N_e








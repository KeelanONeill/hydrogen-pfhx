import numpy as np
import CoolProp.CoolProp as CP
from hydrogen_pfhx import ortho_para_dynamics, hydrogen_thermodynamics


class FluidStream(object):
    # Defines a fluid stream which is an inlet for a conversion HEX.

    # Construct fluid stream
    #TODO - define mass flow rate elsewhere.
    def __init__(self, fluid, mass_flow_rate):
        # define input attributes
        self.fluid = fluid
        self.mass_flow_rate = mass_flow_rate

        # setup abstract state
        self.abstract_state = CP.AbstractState('HEOS', fluid)

        # Environmental variables
        self.temperature = None
        self.pressure = None
        self.phase_fraction = None

        # Derived properties
        self.phase_index = None             # see below
        self.molecular_mass = None
        self.molar_density = None                 # molar density
        self.mass_density = None
        self.viscosity = None               # dynamic
        self.specific_heat_capacity = None  # constant pressure
        self.thermal_conductivity = None
        self.prandtl_number = None
        self.enthalpy = None
        self.entropy = None
        self.saturation_pressure = None
        self.latent_heat_vaporisation = None
        self.molar_flow_rate = None
        self.liquid_fraction = None
        self.vapor_fraction = None

        self.velocity = None
        self.reynolds_number = None
        self.critical_temperature = None
        self.critical_pressure = None

    def update_conditions(self, temperature, pressure):
        self.temperature = temperature
        self.pressure = pressure
        self.abstract_state.update(CP.PT_INPUTS, pressure, temperature)
        # self.abstract_state.specify_phase(CP.iphase_supercritical_liquid)

    def specify_phase(self, phase):
        self.abstract_state.specify_phase(phase)

    def set_mole_fractions(self, z):
        self.mole_fractions = z
        self.abstract_state.set_mole_fractions(z)

    def set_properties(self):
        # Set all key properties for fluid.
        self.molecular_mass = self._calculate_molecular_mass()
        self.molar_density = self._calculate_molar_density()
        self.mass_density = self.molar_density * self.molecular_mass
        self.viscosity = self._calculate_viscosity()
        self.specific_heat_capacity = self._calculate_specific_heat_capacity()
        self.thermal_conductivity = self._calculate_thermal_conductivity()
        self.prandtl_number = self.viscosity * \
            self.specific_heat_capacity/self.thermal_conductivity
        self.enthalpy = self._calculate_enthalpy()
        self.entropy = self._calculate_entropy()
        self.molar_flow_rate = self.mass_flow_rate / self.molecular_mass
        self.critical_temperature = self._calculate_critical_temperature()
        self.critical_pressure = self._calculate_critical_pressure()

    def calculate_velocity(self, cross_sectional_area):
        self.velocity = self.mass_flow_rate / \
            (self.mass_density * cross_sectional_area)

    # Reynolds number

    def calculate_reynolds_number(self, characteristic_length):
        self.reynolds_number = self.mass_density * \
            self.velocity * characteristic_length / self.viscosity

    # base methods for evaluating derived properties (can be modified
    # by subclasses, e.g. Hydrogen with mixed para/ortho)

    def _calculate_molar_density(self):
        # if two-phase - then calculate contribution of each phase
        if self.phase_index == 6:
            # get liquid property
            self.abstract_state.specify_phase(CP.iphase_liquid)
            liquid_density = self.abstract_state.rhomolar()

            # get vapor property
            self.abstract_state.specify_phase(CP.iphase_vapour)
            vapor_density = self.abstract_state.rhomolar()

            # calculate combined property
            rho = self.liquid_fraction * liquid_density + self.vapor_fraction * vapor_density

            # return to two phase state
            self.abstract_state.specify_phase(CP.iphase_twophase)
        else:
            rho = self.abstract_state.rhomolar()
            
        return rho

    def _calculate_viscosity(self):

        # if two-phase - then calculate contribution of each phase
        if self.phase_index == 6:
            # get liquid property
            self.abstract_state.specify_phase(CP.iphase_liquid)
            liquid_viscosity = self.abstract_state.viscosity()

            # get vapor property
            self.abstract_state.specify_phase(CP.iphase_liquid)
            vapor_viscosity = self.abstract_state.viscosity()

            # calculate combined property
            mu = self.liquid_fraction * liquid_viscosity + \
                self.vapor_fraction * vapor_viscosity

            # return to two phase state
            self.abstract_state.specify_phase(CP.iphase_twophase)
        else:
            mu = self.abstract_state.viscosity()
            
        return mu

    def _calculate_specific_heat_capacity(self):
        if self.phase_index == 6:
            # get liquid property
            self.abstract_state.specify_phase(CP.iphase_liquid)
            liquid_cp = self.abstract_state.cpmass()

            # get vapor property
            self.abstract_state.specify_phase(CP.iphase_liquid)
            vapor_cp = self.abstract_state.cpmass()

            # calculate combined property
            cp = self.liquid_fraction * liquid_cp + self.vapor_fraction * vapor_cp

            # return to two phase state
            self.abstract_state.specify_phase(CP.iphase_twophase)
        else:
            cp = self.abstract_state.cpmass()

        return cp

    def _calculate_thermal_conductivity(self):
        if self.phase_index == 6:
            # get liquid property
            self.abstract_state.specify_phase(CP.iphase_liquid)
            liquid_conductivity = self.abstract_state.conductivity()

            # get vapor property
            self.abstract_state.specify_phase(CP.iphase_liquid)
            vapor_conductivity = self.abstract_state.conductivity()

            # calculate combined property
            k = self.liquid_fraction * liquid_conductivity + \
                self.vapor_fraction * vapor_conductivity

            # return to two phase state
            self.abstract_state.specify_phase(CP.iphase_twophase)
        else:
            k = self.abstract_state.conductivity()

        return k

    def _calculate_enthalpy(self):
        if self.phase_index == 6:
            # get liquid property
            self.abstract_state.specify_phase(CP.iphase_liquid)
            liquid_enthalpy = self.abstract_state.enthalpy()

            # get vapor property
            self.abstract_state.specify_phase(CP.iphase_liquid)
            vapor_enthalpy = self.abstract_state.enthalpy()

            # calculate combined property
            h = self.liquid_fraction * liquid_enthalpy + \
                self.vapor_fraction * vapor_enthalpy

            # return to two phase state
            self.abstract_state.specify_phase(CP.iphase_twophase)
        else:
            h = self.abstract_state.hmass()

        return h

    def _calculate_entropy(self):
        s = self.abstract_state.smass
        return s

    def _calculate_latent_heat_vaporisation(self):
        # adjust to saturated liquid
        self.abstract_state.update(CP.QT_INPUTS, 1, self.temperature)
        h_v = self.abstract_state.hmass()

        # adjust to saturated gas
        self.abstract_state.update(CP.QT_INPUTS, 0, self.temperature)
        h_l = self.abstract_state.hmass()

        # calculate latent heat of vaporisation
        lhv = h_v - h_l

        # return to initial conditions
        self.abstract_state.update(
            CP.PT_INPUTS, self.pressure, self.temperature)
        return lhv

    # single parameter property
    def _calculate_saturation_pressure(self):
        # adjust to saturated gas
        self.abstract_state.update(CP.QT_INPUTS, 0, self.temperature)
        p_sat = self.abstract_state.p()

        # return to initial conditions
        self.abstract_state.update(
            CP.PT_INPUTS, self.pressure, self.temperature)
        return p_sat

    # trivial properties
    def _calculate_molecular_mass(self):
        m = self.abstract_state.molar_mass()
        return m

    def _calculate_critical_temperature(self):
        Tc = self.abstract_state.T_critical()
        return Tc

    def _calculate_critical_pressure(self):
        Pc = self.abstract_state.p_critical()
        return Pc


class Hydrogen(FluidStream):
    # Hydrogen stream inlet into HEX
    def __init__(self, mass_flow_rate):
        FluidStream.__init__(self, 'Hydrogen', mass_flow_rate)
        self.ortho = hydrogen_thermodynamics.OrthoHydrogen()
        self.para = hydrogen_thermodynamics.ParaHydrogen()
        self.normal = hydrogen_thermodynamics.NormalHydrogen()

    def set_properties(self):
        # Only set the densities once per update in conditions. Need
        # to set before other properties so they can be used, but also
        # need to determine rho guess first.
        rho_guess = self.abstract_state.rhomolar()
        self.ortho_density = self.ortho.calculate_density(
            self.pressure, self.temperature, rho_guess)
        self.para_density = self.para.calculate_density(
            self.pressure, self.temperature, rho_guess)
        self.normal_density = self.normal.calculate_density(
            self.pressure, self.temperature, rho_guess)

        FluidStream.set_properties(self)

    def update_composition(self, xp):
        self.xp = xp

    def get_xp(self):
        xp = self.xp
        return xp

    def get_xo(self):
        xo = 1.0 - self.xp
        return xo

    def get_xp_equil(self):
        (xpe, _) = ortho_para_dynamics.para_ortho_equilibrium(self.temperature)
        return xpe

    def get_xo_equil(self):
        xpo = 1.0 - self.get_xp_equil()
        return xpo

    def get_heat_of_conversion(self):
        ortho_enthalpy = self.ortho.enthalpy(
            self.ortho_density, self.temperature)
        para_enthalpy = self.para.enthalpy(self.para_density, self.temperature)
        delta_h = para_enthalpy - ortho_enthalpy
        return delta_h

    # linear mixing rule for bulk hydrogen density from ortho/para
    # combo.

    def _calculate_molar_density(self):
        rho = (1-self.xp)*self.ortho_density + self.xp * \
            self.para_density        # molar density (mol/m3)
        return rho

    def _calculate_thermal_conductivity(self):
        viscosity_contributions = self.abstract_state.viscosity_contributions()
        mu_background = viscosity_contributions['residual'] - \
            viscosity_contributions['critical']

        # para k
        ph_k, _ = self.para.thermal_conductivity(
            self.para_density, self.temperature, mu_background)

        # Normal-hydrogen
        nh_k, _ = self.normal.thermal_conductivity(
            self.normal_density, self.temperature, mu_background)

        # Linear interp/extrap of k
        k = nh_k + (ph_k - nh_k) * (0.75 - self.get_xo())/0.75
        return k

    def _calculate_specific_heat_capacity(self):
        # para cp
        ph_cp = self.para.specific_heat_constant_pressure(
            self.para_density, self.temperature)

        # ortho cp
        oh_cp = self.ortho.specific_heat_constant_pressure(
            self.ortho_density, self.temperature)

        # Linear interp of cp
        # molar specific heat capacity (J/mol K)
        cp_molar = (1-self.xp)*oh_cp + self.xp*ph_cp
        cp = cp_molar / self.molecular_mass  # convert to cp mass (J/kg K)
        return cp

    def _calculate_viscosity(self):
        v = self.normal.viscosity(
            self.normal_density, self.temperature, self.molecular_mass)
        return v

    def _calculate_enthalpy(self):
        # para h
        ph_h = self.para.enthalpy(self.para_density, self.temperature)

        # ortho h
        oh_h = self.ortho.enthalpy(self.ortho_density, self.temperature)

        # Linear interp of h
        # molar specific heat capacity (J/mol)
        h_molar = (1-self.xp)*oh_h + self.xp*ph_h
        h = h_molar / self.molecular_mass  # convert to h mass (J/kg)
        return h

    def _calculate_entropy(self):
        # para s
        ph_s = self.para.entropy(self.para_density, self.temperature)

        # ortho s
        oh_s = self.ortho.entropy(self.ortho_density, self.temperature)

        # Linear interp of cp
        # molar specific heat capacity (J/mol)
        s_molar = (1-self.xp)*oh_s + self.xp*ph_s
        s = s_molar / self.molecular_mass  # convert to s mass (J/kg)
        return s

    def _calculate_critical_temperature(self):
        Tc = self.get_xo()*self.ortho.Tc + self.get_xp()*self.para.Tc
        return Tc

    def _calculate_critical_pressure(self):
        Pc = self.get_xo()*self.ortho.Pc + self.get_xp()*self.para.Pc
        return Pc


class HeliumNeon(FluidStream):
    def __init__(self, mass_flow_rate, helium_fraction):
        FluidStream.__init__(self, 'Helium&Neon', mass_flow_rate)
        self.specify_phase(CP.iphase_gas)
        self.set_mole_fractions([helium_fraction, 1 - helium_fraction])

        self.helium_as = CP.AbstractState('HEOS', 'Helium')
        self.neon_as = CP.AbstractState('HEOS', 'Neon')

        # ref temp
        self.T0 = 298.15
        self.mu_0 = np.array([19.8253, 31.7088])*1e-6
        self.lam_0 = np.array([155.0008, 49.1732])*1e-3

        # viscosity coeffs
        self.a_coeffs = np.array([[6.8257552e-1,	6.75404e-1],
                                  [1.4496203e-2,	-2.03522e-2],
                                  [1.1987706e-3,	1.61102e-2],
                                  [-6.7722412e-5,	-4.88074e-3],
                                  [4.9875650e-5,	5.32334e-4],
                                  [-6.1456994e-6,	2.93695e-4],
                                  [1.3189407e-6,	-1.55155e-4],
                                  [-3.7245774e-7,	3.10797e-5],
                                  [1.3671981e-8,	-2.50504e-6],
                                  [5.0354149e-8,	2.74563e-8],
                                  [-1.5714379e-8, 0.0],
                                  [1.4720785e-9, 0.0]])

        # thermal cond coeffs
        self.b_coeffs = np.array([[6.8192175e-1,	6.76478e-1],
                                  [1.4441872e-2,	-2.13734e-2],
                                  [1.2138429e-3,	1.63523e-2],
                                  [-7.4912205e-5,	-4.79402e-3],
                                  [5.2123986e-5,	4.23959e-4],
                                  [-6.0795764e-6,	3.35013e-4],
                                  [8.6135146e-7,	-1.53714e-4],
                                  [-2.6311453e-7,	2.45053e-5],
                                  [6.8367328e-8,	-5.03701e-7],
                                  [1.6608814e-8,	-1.67000e-7],
                                  [-9.0525341e-9, 0.0],
                                  [9.9986305e-10, 0.0]])

    def _calculate_viscosity(self):
        T = self.temperature
        inds = np.arange(start=1, stop=self.a_coeffs.shape[0]+1)
        mu_vals = np.zeros((2,))
        for f_idx in (0, 1):
            a_vals = self.a_coeffs[:, f_idx]
            terms = a_vals * np.log(T / self.T0) ** inds
            r_mu_gas = np.exp(np.sum(terms))
            mu_vals[f_idx] = r_mu_gas * self.mu_0[f_idx]

        mu = np.matmul(self.mole_fractions, mu_vals)
        return mu

    def _calculate_thermal_conductivity(self):
        T = self.temperature
        inds = np.arange(start=1, stop=self.b_coeffs.shape[0]+1)
        lam_vals = np.zeros((2,))
        for f_idx in (0, 1):
            b_vals = self.b_coeffs[:, f_idx]
            terms = b_vals * np.log(T / self.T0) ** inds
            r_lam_gas = np.exp(np.sum(terms))
            lam_vals[f_idx] = r_lam_gas * self.lam_0[f_idx]

        lam = np.matmul(self.mole_fractions, lam_vals)
        return lam

    def _calculate_critical_temperature(self):
        z = self.mole_fractions
        he_tcrit = self.helium_as.T_critical()
        ne_tcrit = self.neon_as.T_critical()
        Tc = z[0]*he_tcrit + z[1]*ne_tcrit
        return Tc

    def _calculate_critical_pressure(self):
        z = self.mole_fractions
        he_pcrit = self.helium_as.p_critical()
        ne_pcrit = self.neon_as.p_critical()
        Pc = z[0]*he_pcrit + z[1]*ne_pcrit
        return Pc

import numpy as np
import scipy.optimize
import CoolProp.CoolProp as CP

class HydrogenData(object):
    # Relevant equations for calculating hydrogen properties
    # Equations obtained from:
    # Leachman JW, Jacobsen RT, Penoncello SG, Lemmon EW.
    # Fundamental equations of state for parahydrogen, normal
    # hydrogen, and orthohydrogen. J Phys Chem Ref Data
    # 200938:721. https://doi.org/10.1063/1.3160306
    
   # properties
    # Universal gas constant
    R = 8.31446261815324 # m3.Pa / (K.mol)
    
    # EOS index values
    l = 7
    m = 9
    n = 14
    
    # value for limit
    limit_eval = 1e-12
    rho_min = 1e-12
    
    # viscosity params
    v_ai = [2.09630e-1,	-4.55274e-1,	1.43602e-1,	-3.35325e-2,	2.76981e-3]
    v_bi = [-0.187,	2.4871,	3.7151,	-11.0972,	9.0965,	-3.8292,	0.5166]
    v_ci = [6.43449673,	4.56334068e-2,	2.32797868e-1,	9.58326120e-1,	1.27941189e-1,	3.63576595e-1]
    
    # solving opts
    
    
    def __init__(self):
        # ref pressure
        self.p_ref = 101325 # Pa
        
        # initialise the reference temperature
        # t_guess = CP.PropsSI('T','P',self.p_ref,'Q',0,'Hydrogen')
        # self.t_ref = fzero(@(T) self.vapour_pressure(T) - self.p_ref, t_guess, self.opt)
        self.t_ref = 1000
                    
        # initialise the reference enthalpy
        rho_guess = CP.PropsSI('Dmolar','P',self.p_ref, 'T',self.t_ref,'Hydrogen')
        rho = self.calculate_density(self.p_ref, self.t_ref, rho_guess)
        self.h_ref = self.absolute_enthalpy(rho, self.t_ref)
                
    def pressure(self, rho, T):
        d_alphar_delta = self.delta_diff_alphar_delta(rho, T)
        p = rho*self.R*T * (1 + d_alphar_delta)
        return p
    
    def Z(self, rho, T):
        d_alphar_delta = self.delta_diff_alphar_delta(rho, T)
        z = 1 + d_alphar_delta
        return z
    
    # Correlation from Muzny et al.
    def viscosity(self, rho, T, M):
        # Dilute gas viscosity contribution
        omega = 0.297 # nm
        e_kb = 30.41 # K
        T_star = T/e_kb
        S_star = np.exp(np.sum(self.v_ai*np.log(T_star)**np.arange(start=0,stop=len(self.v_ai))))
        mu_0 = 0.021357*(M*1e3*T)**0.5/(omega**2.0*S_star)*1e-6
        
        # initial density contribution
        B_mu_star = np.sum(self.v_bi*T_star**(-1*np.arange(start=0,stop=len(self.v_bi))))
        B_mu = B_mu_star * omega **3
        mu_1 = B_mu * mu_0
        mu_init = mu_1 * rho * M
        
        # additional contribution (incorporates both mu_h & mu_c, which
        # are not segregated in Muzny et al.)
        # only use normal hydrogen Tc as that is what is used in paper.
        rho_sc = 90.5 # kg/m3
        Tc = 33.145 # K
        rho_r = rho * M / rho_sc
        T_r = T / Tc
        
        mu_add = 1e-6 * self.v_ci[0] * rho_r**2.0 * np.exp(self.v_ci[1]*T_r + self.v_ci[2]/T_r + self.v_ci[3]*rho_r**2/(self.v_ci[4] + T_r) + self.v_ci[5]*rho_r**6)
        
        mu = mu_0 + mu_init + mu_add
        return mu
    
    def enthalpy(self, rho, T):
        h = self.absolute_enthalpy(rho, T) - self.h_ref
        return h
    
    def entropy(self, rho, T):
        s = self.R * (self.tau_diff_alpha0_tau(T) + self.tau_diff_alphar_tau(rho, T) - self.ideal_helmholtz_energy(rho, T) - self.residual_helmholtz_energy(rho, T))
        return s
    
    # Enthalpy will be in J/mol
    def absolute_enthalpy(self, rho, T):
        h = self.R*T * (1 + self.tau_diff_alpha0_tau(T) + self.tau_diff_alphar_tau(rho, T) + self.delta_diff_alphar_delta(rho, T))
        return h
    
    def specific_heat_constant_volume(self, rho, T):
        tau2_d2_alpha0_dtau2 = self.tau2_diff2_alpha0_difftau2(T)
        tau2_d2_alphar_dtau2 = self.tau2_diff2_alphar_difftau2(rho, T)
        cv = -self.R * (tau2_d2_alpha0_dtau2 + tau2_d2_alphar_dtau2)
        return cv
    
    def specific_heat_constant_pressure(self, rho, T):
        cv = self.specific_heat_constant_volume(rho, T)
        numerator = 1 + self.delta_diff_alphar_delta(rho, T) - self.delta_tau_diff2_alphar_diff_delta_tau(rho,T)
        denominator = 1 + 2.0*self.delta_diff_alphar_delta(rho, T) + self.delta2_diff2_alphar_diffdelta2(rho,T)
        cp = cv + self.R*numerator**2/denominator
        return cp
    
    def speed_of_sound(self, rho, T):
        u = self.R*T * (self.tau_diff_alpha0_tau(T) + self.tau_diff_alphar_tau(rho, T))
        return u

    def pressure_error(self, rho, T, P):
        pe = self.pressure(rho, T) - P
        return pe
    
    def calculate_density(self, P, T, rho_guess):
        rho = scipy.optimize.fsolve(self.pressure_error,  x0 = rho_guess, args=(T, P), xtol=1e-6)
        return rho
         
    # Correlation & data from Assael et al.
    def thermal_conductivity(self, rho, T, mu_background):
        if type(self) == 'modelling.hydrogen_thermodynamics.OrthoHydrogen':
            print('thermal conductivity data is not available for ortho-hydrogen - define as mixture of para/normal')
        else:
            # reduced temperature & density
            rho_r = rho / self.rho_c
            T_r = T / self.Tc
            
            # dilute gas thermal conductivity
            ai_m = np.arange(len(self.tc_A1))
            ai_n = np.arange(len(self.tc_A2))
            lambda_0 = np.sum(self.tc_A1 * T_r**ai_m) / np.sum(self.tc_A2 * T_r**ai_n)
            
            # excess thermal conductivity
            bi = np.arange(len(self.tc_B1))
            lambda_excess = np.sum((self.tc_B1 + self.tc_B2 * T_r) * rho_r**bi)
            
            # empirical critical enhancement
            model = 'olchowy_sengers'
            if model == 'none':
                lambda_c = 0
            elif model == 'empirical':
                delta_Tc = T_r - 1
                delta_rho_r = rho_r - 1
                lambda_c = self.tc_C[0] / (self.tc_C[1] + abs(delta_Tc)) * np.exp(-(self.tc_C[2]*delta_rho_r)**2)
            elif model == 'olchowy_sengers':
                # Olchowy and Sengers model
                cp = self.specific_heat_constant_pressure(rho, T)
                cv = self.specific_heat_constant_volume(rho, T)
                
                # constants
                kb = 1.38064852e-23 # m2 kg s-2 K-1
                RD = 1.01
                nu = 0.63
                gamma = 1.2415
                GAMMA = 0.172
                xi_0 = 1.5e-10
                T_ref = 1.5 * self.Tc
                
                # Eq 7, 6 & 5 (Assael et al)
                del_psi = self.R *self.Tc * rho / self.rho_c * (self.diff_rho_pressure(rho, T) - T_ref/T * self.diff_rho_pressure(rho, T_ref))
                min_val = 1e-16
                xi = xi_0 * (GAMMA**(-1) * np.max([min_val,del_psi])) ** (nu/gamma)
                y = self.qd*xi
                omega_0 = 2/np.pi * (1 - np.exp(-1/((y)**(-1.0) + (y*self.rho_c/rho)**2/3)))
                omega = 2/np.pi * ((cp - cv)/cp * np.arctan(y) + cv/cp*y)
                
                if xi > 0:
                    lambda_c = rho * cp * RD * kb * T /(6*np.pi*mu_background*xi) * (omega - omega_0)
                else:
                    lambda_c = 0
                
                lambda_c = np.max([0, lambda_c])
            
            
            lambda_val = lambda_0 + lambda_excess + lambda_c
    
        return lambda_val, lambda_c
    
    # def for specific heat capacity (Eq 25)
    def ideal_specific_heat_capacity(self, T):
        cp = self.R * (2.5 + np.sum(self.uk * (self.vk/T)**2.0 * np.exp(self.vk/T) / (np.exp(self.vk/T) - 1)**2))
        return cp
    
    # def for ideal component of Helmholtz energy (Eq 31)
    def ideal_helmholtz_energy(self, rho, T):
        delta = rho / self.rho_c
        tau = self.Tc / T
        alpha_0 = np.log(delta) + 1.5 * np.log(tau) + self.ak[0] + self.ak[1]*tau +\
            np.sum(self.ak[2:] * np.log(1 - np.exp(self.bk[2:]*tau)))
        return alpha_0
    
    # def for residual component of Helmholtz energy (Eq 32)
    def residual_helmholtz_energy(self, rho, T):
        delta = rho / self.rho_c
        tau = self.Tc / T
        main_terms = self.Ni_eos * (delta ** self.di) * (tau ** self.ti)
        gaussian_terms = self.phi_i *(delta - self.D_i)**2.0 + self.beta_i * (tau - self.gamma_i)**2
        alpha_r = np.sum(main_terms[:self.l]) + \
            np.sum(main_terms[self.l:self.m]*np.exp(-(delta) ** self.pi[self.l:self.m])) + \
            np.sum(main_terms[self.m:self.n]*np.exp(-gaussian_terms))
        return alpha_r
    
    # combined helmholtz energy (Eq 20)
    def reduced_helmholtz_energy(self, rho, T):
        alpha = self.ideal_helmholtz_energy(rho, T) + self.residual_helmholtz_energy(rho, T)
        return alpha
    
    # final Helmholtz energy (not reduced) (Eq 17)
    def helmholtz_energy(self, rho, T):
        a = self.reduced_helmholtz_energy(rho, T) * self.R * T
        return a
    
    def vapour_pressure(self, T):
        tau = self.Tc / T
        theta = 1 - T / self.Tc
        p_omega = self.Pc * np.exp(tau * np.sum(self.Ni_vp * theta ** self.ki))
        return p_omega
    
    def second_virial_coefficient(self, T):
        # syms rho
        f = 1 / self.rho_c*self.delta_diff_alphar_delta(self.rho_min, T) # @(rho) 
        B = f #(self.limit_eval) #eval(limit(f, rho, 0,'right'))
        return B
    
    def third_virial_coefficient(self, T):
        # syms rho
        f = 1 / self.rho_c()**2.0 * self.delta2_diff2_alphar_diffdelta2(self.rho_min, T) # @(rho)
        C = f #(self.limit_eval) #eval(limit(f, rho, 0,'right'))
        return C
    
    def diff_rho_pressure(self, rho, T):
        # at constant T
        drho_dp = 1/self.diff_pressure_rho(rho, T)
        return drho_dp
    
    def diff_pressure_rho(self, rho, T):
        # (at constant T)
        dp_drho = self.R * T * (1 + 2.0*self.delta_diff_alphar_delta(rho, T) + self.delta2_diff2_alphar_diffdelta2(rho, T))
        return dp_drho
    
    def delta_diff_alphar_delta(self, rho, T):
        delta = rho / self.rho_c
        tau = self.Tc / T
        main_terms = self.Ni_eos * delta ** self.di * tau ** self.ti
        extra_terms = (self.di - self.pi * delta**self.pi)
        gaussian_terms = self.phi_i *(delta - self.D_i)**2.0 + self.beta_i * (tau - self.gamma_i)**2
        extra_gaussian_terms = self.di[self.m:self.n] - 2.0*delta*self.phi_i*(delta - self.D_i)
        d_alphar_delta = np.sum(self.di[:self.l]*main_terms[:self.l]) + \
            np.sum(main_terms[self.l:self.m]*np.exp(-(delta) ** self.pi[self.l:self.m])*extra_terms[self.l:self.m]) + \
            np.sum(main_terms[self.m:self.n]*np.exp(-gaussian_terms)*extra_gaussian_terms)
        return d_alphar_delta
    
    def tau_diff_alpha0_tau(self, T):
        tau = self.Tc / T
        d_alpha0_tau = 1.5 + tau*self.ak[1] +\
            tau*np.sum(self.ak[2:] * -1.0*self.bk[2:]*np.exp(self.bk[2:]*tau) / (1 - np.exp(self.bk[2:]*tau)))
        return d_alpha0_tau
    
    def tau_diff_alphar_tau(self, rho, T):
        delta = rho / self.rho_c
        tau = self.Tc / T
        main_terms = self.Ni_eos * delta ** self.di * tau ** self.ti
        extra_terms = self.ti[self.l:self.m]
        gaussian_terms = self.phi_i *(delta - self.D_i)**2.0 + self.beta_i * (tau - self.gamma_i)**2
        extra_gaussian_terms = self.ti[self.m:self.n] - 2.0*tau*self.beta_i*(tau - self.gamma_i)
        d_alphar_tau = np.sum(self.ti[:self.l]*main_terms[:self.l]) + \
            np.sum(main_terms[self.l:self.m]*np.exp(-(delta) ** self.pi[self.l:self.m])*extra_terms) + \
            np.sum(main_terms[self.m:self.n]*np.exp(-gaussian_terms)*extra_gaussian_terms)
        return d_alphar_tau
    
    def tau2_diff2_alpha0_difftau2(self, T):
        tau = self.Tc / T
        eff_bk = np.array(self.bk[2:])
        d2_alpha0_dtau2 = -1.5 - tau**2.0 *np.sum(self.ak[2:] * eff_bk**2.0 * np.exp(eff_bk*tau) * ((np.exp(eff_bk*tau) - 1.0)**(-2.0)))
        return d2_alpha0_dtau2
    
    def tau2_diff2_alphar_difftau2(self, rho, T):
        delta = rho / self.rho_c
        tau = self.Tc / T
        main_terms = self.Ni_eos * delta ** self.di * tau ** self.ti
        ti_array = np.array(self.ti)
        extra_terms = ti_array * (np.array(self.ti) - 1.0)
        gaussian_terms = self.phi_i *(delta - self.D_i)**2.0 + self.beta_i * (tau - self.gamma_i)**2
        extra_gaussian_terms = (ti_array[self.m:self.n] - 2.0 * np.array(self.beta_i) * tau*(tau - np.array(self.gamma_i)))**2.0 - \
            ti_array[self.m:self.n] - 2.0*tau**2.0 *np.array(self.beta_i)
        d2_alphar_dtau2 = np.sum(extra_terms[:self.l]*main_terms[:self.l]) + \
            np.sum(main_terms[self.l:self.m]*np.exp(-(delta) ** self.pi[self.l:self.m])*extra_terms[self.l:self.m]) + \
            np.sum(main_terms[self.m:self.n]*np.exp(-gaussian_terms)*extra_gaussian_terms)
        return d2_alphar_dtau2
    
    def delta_tau_diff2_alphar_diff_delta_tau(self,rho,T):
        delta = rho / self.rho_c
        tau = self.Tc / T
        main_terms = self.Ni_eos * delta ** self.di * tau ** self.ti
        li = self.pi[self.l:self.m]
        extra_terms = (self.di[self.l:self.m] - li * (delta) ** li) * self.ti[self.l:self.m]
        gaussian_terms = self.phi_i *(delta - self.D_i)**2.0 + self.beta_i * (tau - self.gamma_i)**2.0
        extra_gaussian_terms = (self.di[self.m:self.n] - 2.0 * np.array(self.phi_i) * delta *(delta - np.array(self.D_i))) * \
            (self.ti[self.m:self.n] - 2.0 * np.array(self.beta_i) * tau*(tau - np.array(self.gamma_i)))
        d2_alphar_delta_tau = np.sum(np.array(self.di[:self.l]) * np.array(self.ti[:self.l]) * main_terms[:self.l]) + \
            np.sum(main_terms[self.l:self.m]*np.exp(-(delta) ** li)*extra_terms) + \
            np.sum(main_terms[self.m:self.n]*np.exp(-gaussian_terms)*extra_gaussian_terms)
        return d2_alphar_delta_tau
    
    def delta2_diff2_alphar_diffdelta2(self,rho,T):
        delta = rho / self.rho_c
        tau = self.Tc / T
        main_terms = self.Ni_eos * delta ** self.di * tau ** self.ti
        li = self.pi[self.l:self.m]
        di_array = np.array(self.di)
        extra_terms = (di_array[self.l:self.m] - li * delta ** li) * \
            (di_array[self.l:self.m] - 1.0 - li * delta ** li) - \
            li**2.0 * delta ** li
        gaussian_terms = self.phi_i *(delta - self.D_i)**2.0 + self.beta_i * (tau - self.gamma_i)**2.0
        extra_gaussian_terms = (self.di[self.m:self.n] - 2.0 * self.phi_i * delta *(delta - self.D_i))**2.0 - \
            self.di[self.m:self.n] - 2.0*delta**2.0 *self.phi_i
        d2_alphar_ddelta2 = np.sum(self.di[:self.l] * (self.di[:self.l] - 1) * main_terms[:self.l]) + \
            np.sum(main_terms[self.l:self.m]*np.exp(-(delta) ** li)*extra_terms) + \
            np.sum(main_terms[self.m:self.n]*np.exp(-gaussian_terms)*extra_gaussian_terms)
            
        return d2_alphar_ddelta2
    
    
class NormalHydrogen(HydrogenData):
    # property data for Normal hydrogen (75# para, 25# ortho)
    # Data obtained from:
    # Leachman JW, Jacobsen RT, Penoncello SG, Lemmon EW.
    # Fundamental equations of state for parahydrogen, normal
    # hydrogen, and orthohydrogen. J Phys Chem Ref Data
    # 200938:721. https://doi.org/10.1063/1.3160306
    def __init__(self):
        # specific heat capacity coefficient data
        self.uk = [1.616, -0.4117, -0.792, 0.758, 1.217]
        self.vk = [531, 751, 1989, 2484, 6859]

        # ideal gas heat capacity coefficients
        self.ak = np.array([-1.457985648,	1.888076782,	1.616,	-0.4117,	-0.792,	0.758,	1.217])
        self.bk = np.array([np.nan,	np.nan,	-16.02051591,	-22.6580178,	-60.00905114,	-74.94343038,	-206.9392065])
        
        # Polynomial terms for hydrogen EOS
        self.Ni_eos = [-6.93643,	0.01,	2.11101,	4.52059,	0.732564,	-1.34086,	0.130985,	-0.777414,	0.351944,	-0.0211716,	0.0226312,	0.032187,	-0.0231752,	0.0557346]
        self.ti = np.array([0.6844,	1.0,	0.989,	0.489,	0.803,	1.1444,	1.409,	1.754,	1.311,	4.187,	5.646,	0.791,	7.249,	2.986])
        self.di = np.array([1,	4,	1,	1,	2,	2,	3,	1,	3,	2,	1,	3,	1,	1])
        self.pi = np.array([0,	0,	0,	0,	0,	0,	0,	1,	1,	np.nan, np.nan, np.nan, np.nan, np.nan])
        
        # Gaussian terms for hydrogen EOS
        self.phi_i = np.array([1.685,	0.489,	0.103,	2.506,	1.607])
        self.beta_i = np.array([0.171,	0.2245,	0.1304,	0.2785,	0.3967])
        self.gamma_i = np.array([0.7164,	1.3444,	1.4517,	0.7204,	1.5445])
        self.D_i = np.array([1.506,	0.156,	1.736,	0.67,	1.662])
        
        # Critical properties
        self.Tc = 33.145 # K
        self.Pc = 1.2964e6 # Pa
        self.rho_c = 15.508e3 # mol/m3
        
        # Vapour pressure properties
        self.Ni_vp = [-4.89789,	0.988558,	0.349689,	0.499356]
        self.ki = [1, 1.5, 2, 2.85]
        
        # Thermal conductivity parameters (from Assael et al.)
        self.tc_A1 = np.array([-3.40976E-01,	4.58820E+00,	-1.45080E+00,	3.26394E-01,	3.16939E-03,	1.90592E-04,	-1.13900E-06])
        self.tc_A2 = np.array([1.38497E+02,	-2.21878E+01,	4.57151E+00,	1.00000E+00])
							
        self.tc_B1 = np.array([3.63081E-02,	-2.07629E-02,	3.14810E-02,	-1.43097E-02,	1.74980E-03])
        self.tc_B2 = np.array([1.83370E-03,	-8.86716E-03,	1.58260E-02,	-1.06283E-02,	2.80673E-03])
							
        self.tc_C = [6.24e-4, -2.58e-7, 0.837]
        
        self.qd = (4e-10)**(-1.0)
        
        # call superclass init
        HydrogenData.__init__(self)


class OrthoHydrogen(HydrogenData):
    # property data for Orthohydrogen
    # Data obtained from:
    # Leachman JW, Jacobsen RT, Penoncello SG, Lemmon EW.
    # Fundamental equations of state for parahydrogen, normal
    # hydrogen, and orthohydrogen. J Phys Chem Ref Data
    # 200938:721. https://doi.org/10.1063/1.3160306
    def __init__(self):
        # specific heat capacity coefficient data
        self.uk = [2.54151, -2.3661, 1.00365, 1.22447]
        self.vk = [856, 1444, 2194, 6968]

        # ideal gas heat capacity coefficients
        self.ak = np.array([-1.4675442336, 1.8845068862, 2.54151, -2.3661, 1.00365, 1.22447])
        self.bk = np.array([np.nan, np.nan, -25.7676098736, -43.4677904877, -66.0445514750, -209.7531607465])
        
        # Polynomial terms for hydrogen EOS
        self.Ni_eos = [-6.83148,		0.01,		2.11505,		4.38353,		0.211292,		-1.00939,		0.142086,	-0.87696,	0.804927,	-0.710775,	0.0639688,	0.0710858,	-0.087654,	0.647088]
        self.ti = np.array([0.7333,	1.0,	1.1372,	0.5136,	0.5638,	1.6248,	1.829,	2.404,	2.105,	4.1,	7.658,	1.259,	7.589,	3.946])
        self.di = np.array([1,	4,	1,	1,	2,	2,	3,	1,	3,	2,	1,	3,	1,	1])
        self.pi = np.array([0,	0,	0,	0,	0,	0,	0,	1,	1,	np.nan, np.nan, np.nan, np.nan, np.nan])
        
        # Gaussian terms for hydrogen EOS
        self.phi_i = np.array([1.169,	0.894,	0.04,	2.072,	1.306])
        self.beta_i = np.array([0.4555,	0.4046,	0.0869,	0.4415,	0.5743])
        self.gamma_i = np.array([1.5444,	0.6627,	0.763,	0.6587,	1.4327])
        self.D_i = np.array([0.6366,	0.3876,	0.9437,	0.3976,	0.9626])
        
        # Critical properties
        self.Tc = 33.22 # K
        self.Pc = 1.31065*1e6 # Pa
        self.rho_c = 15.445e3 # mol/m3
        
        # Vapour pressure properties
        self.Ni_vp = [-4.88684,	1.0531,	0.856947,	-0.185355]
        self.ki = [1,	1.5,	2.7,	6.2]
        
        # call superclass init
        HydrogenData.__init__(self)


class ParaHydrogen(HydrogenData):
    # property data for Parahydrogen
    # Data obtained from:
    # Leachman JW, Jacobsen RT, Penoncello SG, Lemmon EW.
    # Fundamental equations of state for parahydrogen, normal
    # hydrogen, and orthohydrogen. J Phys Chem Ref Data
    # 200938:721. https://doi.org/10.1063/1.3160306
    def __init__(self):
        # specific heat capacity coefficient data
        self.uk = [4.30256, 	13.0289, 	-47.7365, 	50.0013, 	-18.6261, 	0.993973, 	0.536078]
        self.vk = [499, 	826.5, 	970.8, 	1166.2, 	1341.4, 	5395, 	10185]

        # ideal gas heat capacity coefficients
        self.ak = np.array([-1.4485891134, 1.884521239, 	4.30256, 	13.0289, 	-47.7365, 	50.0013, 	-18.6261, 	0.993973, 	0.536078])
        self.bk = np.array([np.nan, np.nan, -15.14967515, 	-25.09259821, 	-29.47355638, 	-35.40591414, 	-40.72499848, 	-163.79258, 	-309.2173174])
        
        # Polynomial terms for hydrogen EOS
        self.Ni_eos = [-7.33375, 	0.01, 	2.60375, 	4.66279, 	0.68239, 	-1.47078, 	0.135801, 	-1.05327, 	0.328239, 	-0.0577833, 	0.0449743, 	0.0703464, 	-0.0401766, 	0.11951]
        self.ti = np.array([0.6855, 	1.0, 	1.0, 	0.489, 	0.774, 	1.133, 	1.386, 	1.619, 	1.162, 	3.96, 	5.276, 	0.99, 	6.791, 	3.19])
        self.di = np.array([1, 	4, 	1, 	1, 	2, 	2, 	3, 	1, 	3, 	2, 	1, 	3, 	1, 	1])
        self.pi = np.array([0, 	0, 	0, 	0, 	0, 	0, 	0, 	1, 	1, np.nan, np.nan, np.nan, np.nan, np.nan])
        
        # Gaussian terms for hydrogen EOS
        self.phi_i = np.array([1.7437, 	0.5516, 	0.0634, 	2.1341, 	1.777])
        self.beta_i = np.array([0.194, 	0.2019, 	0.0301, 	0.2383, 	0.3253])
        self.gamma_i = np.array([0.8048, 	1.5248, 	0.6648, 	0.6832, 	1.493])
        self.D_i = np.array([1.5487, 	0.1785, 	1.28, 	0.6319, 	1.7104])
        
        # Critical properties
        self.Tc = 32.938 # K
        self.Pc = 1.2858*1e6 # Pa
        self.rho_c = 15.538e3 # mol/m3
        
        # Vapour pressure properties
        self.Ni_vp = [-4.87767, 	1.03359, 	0.82668, 	-0.129412]
        self.ki = [1, 	1.5, 	2.65, 	7.4]
        
        # Thermal conductivity parameters (from Assael et al.)
        self.tc_A1 = np.array([-1.24500E+00, 	3.10212E+02, 	-3.31004E+02, 	2.46016E+02, 	-6.57810E+01, 	1.08260E+01, 	-5.19659E-01, 	1.43979E-02])
        self.tc_A2 = np.array([1.42304E+04, 	-1.93922E+04, 	1.58379E+04, 	-4.81812E+03, 	7.28639E+02, 	-3.57365E+01, 	1.00000E+00])

        self.tc_B1 = np.array([2.65975E-02, 	-1.33826E-03, 	1.30219E-02, 	-5.67678E-03, 	-9.23380E-05])
        self.tc_B2 = np.array([-1.21727E-03, 	3.66663E-03, 	3.88715E-03, 	-9.21055E-03, 	4.00723E-03])

        self.tc_C = [3.57e-4, -2.46e-2, 	0.2]
        
        self.qd = (5e-10)**(-1.0)
        
        # call superclass init
        HydrogenData.__init__(self)

    
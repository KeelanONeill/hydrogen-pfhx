import numpy as np
from hydrogen_pfhx import pressure_models


def aluminium_thermal_conductivity(temperature):
    # Aluminium 3003 thermal conductivity
    # Woodcraft, Adam L. "Predicting the thermal conductivity of aluminium alloys in the cryogenic to room temperature range." Cryogenics 45.6 (2005): 421-431.
    rho_rt_pure = 2.43e-8  # ohm-m
    L0 = 2.45e-8  # W ohm / K^2
    RRR_star = 2.65  # relative RRR for 3003
    rho_0 = rho_rt_pure / RRR_star
    beta = rho_0 / L0

    P = np.array([4.716e-8, 2.446, 623.6, -0.16, 130.9, 2.5, 0.8168])

    W0 = beta / temperature
    Wi_a = P[0]*temperature**P[1]
    Wi_b = (1 + P[0]*P[2] * temperature**(P[1] + P[3])
            * np.exp(-((P[4]/temperature)**P[5])))**(-1)
    Wi_c = -5e-4 * np.log(temperature/330) * np.exp(-(np.log(temperature/380)/0.6)**2) - \
        1.3e-3 * np.log(temperature/110) * \
        np.exp(-(np.log(temperature/94)/0.5)**2)
    Wi = Wi_a * Wi_b + Wi_c
    Wi0 = P[6] * Wi*W0/(Wi + W0)
    k = 1 / (W0 + Wi + Wi0)
    return k


def stainless_steel_thermal_conductivity(T):
    # https://trc.nist.gov/cryogenics/materials/304Stainless/304Stainless_rev.htm
    a = -1.4087
    b = 1.3982
    c = 0.2543
    d = -0.6260
    e = 0.2334
    f = 0.4256
    g = -0.4658
    h = 0.1650
    i = -0.0199
    log10_y = a+b*(np.log10(T)) + c*(np.log10(T)) ** 2 + d*(np.log10(T)) ** 3 + e*(np.log10(T)) ** 4 + \
        f*(np.log10(T)) ** 5 + g*(np.log10(T)) ** 6 + \
        h*(np.log10(T)) ** 7 + i*(np.log10(T)) ** 8
    y = 10**(log10_y)
    return y


def gnielinski_equation(f_darcy, Re, Pr, Pr_w, aspect_ratio):
    K = (Pr / Pr_w)**0.11
    if (Pr < 0.5) | (Pr > 2000):
        print('Current Prandtl number (%.2e) is outside correlation limits', Pr)

    if (Re < 2000) | (Re > 5e6):  # edited to 2000 to allow use in transition zone (with interpolation)
        print('Current Reynolds number (%.2e) is outside correlation limits', Re)

    numerator = (f_darcy/8)*Re*Pr
    denominator = 1 + 12.7*(f_darcy/8)**0.5 * (Pr**(2/3)-1)
    correction_term = 1 + (aspect_ratio)**(2/3)

    Nu = numerator / denominator * correction_term * K
    return Nu


def laminar_nusselt_correlation(aspect_ratio):
    # Fundamentals of Heat Exchanger Design
    # Shah, Ramesh K; Sekulic, Dusan P
    # Table 7.4 for Rectangular geometry, Nu_T correlation
    Nu = 7.451*(1-2.61*aspect_ratio + 4.97 * aspect_ratio**2 - 5.119 *
                aspect_ratio**3 + 2.702*aspect_ratio**4 - 0.548*aspect_ratio**5)
    return Nu

# https://www.sciencedirect.com/science/article/pii/S0017931013003207#e0035


def gnielinski_laminar(Re, Pr, d, L):
    Nu_mT1 = 3.66
    Nu_mT2 = 1.615 * (Re * Pr * (d/L))**(1/3)
    Nu_mT3 = (2 / (1+22*Pr))**(1/6) * (Re * Pr * d/L)**0.5
    Nu_mT = (Nu_mT1**3 + 0.7**3 + (Nu_mT2 - 0.7)**3 + Nu_mT3**3)**(1/3)
    return Nu_mT


def full_gnielinski(Re, Pr, d, L, eta, Pr_w):
    # Nusselt number correlations
    aspect_ratio = d/L
    if Re < 0:
        print('Cannot have negative Reynolds')
    elif Re <= 2300:
        Nu_h = gnielinski_laminar(Re, Pr, d, L)
    elif (Re > 2300) & (Re < 4000):
        f_darcy = pressure_models.ergun_equation(eta, Re)
        Nu_lam = gnielinski_laminar(Re, Pr, d, L)

        Nu_turb = gnielinski_equation(f_darcy, Re, Pr, Pr_w, aspect_ratio)
        phi = (Re-2300) / (4000-2300)
        Nu_h = phi*Nu_turb + (1-phi)*Nu_lam
    else:
        f_darcy = pressure_models.ergun_equation(eta, Re)
        Nu_h = gnielinski_equation(f_darcy, Re, Pr, Pr_w, aspect_ratio)

    return Nu_h


def manglik_bergles_heat_transfer_model(Re, alpha, delta, gamma):
    # https://www.sciencedirect.com/science/article/pii/089417779400096Q
    f = 9.6243 * Re**-0.7422 * alpha**-0.1856 * delta**0.3053 * gamma**0.2659 * \
        (1 + 7.669e-8 * Re**4.429 * alpha**0.92 * delta**3.767 * gamma**0.236)**0.1
    j = 0.6522 * Re**-0.5403 * alpha**-0.1541 * delta**0.1499 * gamma**-0.0678 * \
        (1 + 5.269e-5 * Re**1.340 * alpha**0.504 * delta**0.456 * gamma**-1.055)**0.1
    return f, j


def packed_bed_heat_transfer(Dp, Dt, Re_p, Pr):
    # Peters PE, Schiffino RS, Harriott P. Heat transfer in packed tube reactors. Ind Eng Chem Res 1988;27:226e33. https://doi.org/10.1021/ie00074a003
    Nu = 3.8*(Dp/Dt)**0.39*Re_p ^ 0.5*Pr**(1/3)
    return Nu


def zehner_bauer_schlunder_model(k_g, epsilon, **kwargs):
    # Zehner - Bauer - Schlunder model for heat transfer in packed bed.
    omega = 7.26e-3
    void = 1 - epsilon

    k_ge = (1 - epsilon ^ 0.5)*k_g/void

    if 'k_s' in kwargs:
        k_s = kwargs['k_s']  # solid phase thermal conductivity
        B = 1.25*(epsilon/void) ^ (10/9)
        kappa = k_s/k_g
        ombok = 1-B/kappa  # One minus beta over kappa
        gamma = 2/ombok * ((kappa-1)/ombok ^ 2 * B/kappa *
                           np.log(kappa/B) - (B-1)/ombok - 0.5 * (B+1))

        k_se = epsilon ^ 0.5 * (omega*kappa + (1-omega)*gamma) * k_g / epsilon
    return k_ge, k_se

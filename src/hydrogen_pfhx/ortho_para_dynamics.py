import numpy as np


def para_ortho_equilibrium(T):
    # Calculate the equilibrium concentrations of Para- & Ortho- Hydrogen as a
    # function of temperature (T)
    xp = 0.1*(np.exp(-175./T)+0.1)**(-1) - 7.06e-9 * \
        T**3 + 3.42e-6*T**2 - 6.2e-5*T - 0.00227
    xo = 1 - xp
    return xp, xo


def first_order_kinetics(reactant):
    # calculate first order conversion kinetics for ortho > para conversion
    # https://onlinelibrary.wiley.com/doi/abs/10.1002/ceat.201800345
    # extract reactant properties.
    T = reactant.temperature
    xo = reactant.get_xo()
    xo_equil = reactant.get_xo_equil()
    cH2 = reactant.molar_density

    # universal gas constant
    # TODO - import from CP
    R = 8.31446261815324  # J/(mol.K)

    # reactor kinetic constants (from Donaubauer et al.)
    Ea = -336.45                # J/mol
    a = 2.2e-3                 # m3s/mol
    b = -35.11e-3              # s

    # rate constant
    k = np.exp(-Ea/(R*T))/(a*cH2 + b) * 1/(1-xo_equil)  # 1/s

    # first order conversion rate. Note conversion rate defined in terms of
    # production of para-hydrogen as positive.
    conversion_rate = k * cH2 * (xo - xo_equil)  # m3 / (mol.s)

    return conversion_rate

# Ergun equation
def ergun_equation(epsilon, Re_p):
    # dPdx = 150*mu/dp^2*(1-eta)^2/eta^3*vs + 1.75*rho/dp*(1-eta)/eta^3*vs*abs(vs);
    ff = 150/Re_p + 1.75
    return ff

# Hicks equation (pressure drop through packed bed)
def hicks_equation(mu, dp, epsilon, vs, Re_p):
    dPdx = mu*(1-epsilon)**2*vs/(dp**2*epsilon**3) * 6.8 * Re_p**0.8
    return dPdx
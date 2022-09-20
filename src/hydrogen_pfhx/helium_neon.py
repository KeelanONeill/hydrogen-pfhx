import json
import itertools
from collections import namedtuple
import CoolProp.CoolProp as CP
import sys


"""
Use with CoolProp version 6.3.0.
This exact version can be installed from pip with the following command:

pip install CoolProp==6.3.0
"""


def setup_CoolProp(fluids):
    CP.set_config_bool(CP.OVERWRITE_FLUIDS, True)
    CP.set_config_bool(CP.OVERWRITE_BINARY_INTERACTION, True)
    sys.path.append('fluid_properties\\')

    # Read in the fluid files
    for fluid in fluids:
        sys.path.append(fluid+'.json')
        contents = json.load(open(fluid+'.json'))
        CP.add_fluids_as_JSON('HEOS', json.dumps([contents]))

    departure_JSON = [
        {
            "Name": "Helium-Neon",
            "aliases": [],
            "type": "Gaussian+Exponential",
            "BibTeX": "Tkaczuk-2020",
            "Npower": 3,
            "n":       [-4.346849, -0.884378, 0.258416, 3.502188, 0.831330, 2.740495, -1.582230, -0.304897],
            "t":       [1.195,  1.587, 1.434, 1.341, 1.189, 1.169,  0.944,  1.874],
            "d":       [1,  2, 3, 1, 2, 3,  4,  4],
            "l":       [0,  0, 0, 0, 0, 0,  0,  0],
            "eta":     [0.000,  0.000, 0.000, 0.157, 0.931, 0.882,  0.868,  0.543],
            "beta":    [0.000,  0.000, 0.000, 0.173, 1.070, 0.695,  0.862,  0.971],
            "gamma":   [0.000,  0.000, 0.000, 1.310, 1.356, 1.596,  1.632,  0.766],
            "epsilon": [0.000,  0.000, 0.000, 1.032, 1.978, 1.966,  1.709,  0.583]
        },
        {
            "Name": "Helium-Argon",
            "aliases": [],
            "type": "Gaussian+Exponential",
            "BibTeX": "Tkaczuk-2020",
            "Npower": 3,
            "n":        [-2.643654, -0.347501, 0.201207, 1.171326, 0.216379, 0.561370, 0.182570, 0.017879],
            "t":        [1.030,  0.288, 0.572, 1.425, 1.987, 0.024, 1.434, 0.270],
            "d":        [1,  2, 3, 1, 1, 2, 3, 4],
            "l":        [0,  0, 0, 0, 0, 0, 0, 0],
            "eta":      [0.000,  0.000, 0.000, 0.371, 0.081, 0.375, 0.978, 0.971],
            "beta":     [0.000,  0.000, 0.000, 0.320, 1.247, 1.152, 0.245, 1.030],
            "gamma":    [0.000,  0.000, 0.000, 1.409, 1.709, 0.705, 1.162, 0.869],
            "epsilon":  [0.000,  0.000, 0.000, 0.378, 0.741, 0.322, 1.427, 2.088]
        },
        {
            "Name": "Neon-Argon",
            "aliases": [],
            "type": "Gaussian+Exponential",
            "BibTeX": "Tkaczuk-2020",
            "Npower": 3,
            "n":       [-1.039686, 0.593776, -0.186531, -0.223315, 0.160847, 0.405228, -0.264563, -0.033569],
            "t":       [0.723, 1.689,  1.365,  0.201, 0.164, 0.939,  1.690,  1.545],
            "d":       [1, 2,  3,  1, 2, 2,  3,  4],
            "l":       [0, 0,  0,  0, 0, 0,  0,  0],
            "eta":     [0.000, 0.000,  0.000,  1.018, 0.556, 0.221,  0.862,  0.809],
            "beta":    [0.000, 0.000,  0.000,  0.360, 0.373, 0.582,  0.319,  0.560],
            "gamma":   [0.000, 0.000,  0.000,  1.119, 1.395, 1.010,  1.227,  1.321],
            "epsilon": [0.000, 0.000,  0.000,  2.490, 1.202, 2.468,  0.837,  2.144]
        }
    ]
    CP.set_config_bool(CP.NORMALIZE_GAS_CONSTANTS, False)
    CP.set_departure_functions(json.dumps(departure_JSON))
    for pair in itertools.combinations(fluids, 2):
        CP.apply_simple_mixing_rule(pair[0], pair[1], 'Lorentz-Berthelot')


BIPs = namedtuple('BIPS', ['betaT', 'gammaT', 'betaV', 'gammaV', 'Fij'])


def get_AS(fluids):
    BIP = {
        ("Helium", "Neon"):  (0.793, 0.728, 1.142, 0.750, 1.0),
        ("Helium", "Argon"): (1.031, 1.113, 1.048, 0.862, 1.0),
        ("Neon", "Argon"):   (1.033, 0.967, 0.919, 1.035, 1.0),
    }
    AS = CP.AbstractState('HEOS', '&'.join(fluids))
    b = BIPs(*BIP[fluids])
    for k in ['betaT', 'gammaT', 'betaV', 'gammaV', 'Fij']:
        AS.set_binary_interaction_double(0, 1, k, getattr(b, k))
    AS.set_binary_interaction_string(0, 1, 'function', '-'.join(fluids))
    return AS


def do_calc(z):
    setup_CoolProp(['Helium', 'Neon', 'Argon'])
    print(
        'mixture, pressure/Pa, alphar, rho_reducing/(mol*m^{-3}), R/(J/mol*K)')
    for pair, rhomolar in [
        (('Helium', 'Neon'), 1e4),
        (('Helium', 'Argon'), 1e4),
        (('Neon', 'Argon'), 1e4)
    ]:
        AS = get_AS(pair)
        AS.specify_phase(CP.iphase_gas)
        AS.set_mole_fractions(z)
        AS.update(CP.DmolarT_INPUTS, rhomolar, 200)
        print(pair[0] + ' - ' + pair[1] + ':', [AS.p(), AS.alphar(),
              AS.rhomolar_reducing(), AS.gas_constant()])


if __name__ == '__main__':
    do_calc([0.5, 0.5])

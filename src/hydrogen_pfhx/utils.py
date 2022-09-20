
# convert from tpd to kg/s
def tpd_to_kps(mass_flow_rate):
    return mass_flow_rate * 1e3 / (60**2*24)

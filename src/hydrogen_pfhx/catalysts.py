# catalyst object

class Catalyst(object):
    def __init__(self, catalyst_configuration):
        self.solid_fraction = catalyst_configuration['solid_fraction']
        self.particle_diameter = catalyst_configuration['diameter']
        self.density = catalyst_configuration['density']
        self.void_fraction = 1 - catalyst_configuration['solid_fraction']

    def calculate_catalyst_mass(self, reaction_side_volume):
        catalyst_volume = reaction_side_volume * self.solid_fraction
        catalyst_mass = catalyst_volume * self.density
        return catalyst_mass

    # correlation for T.C. of hematite (Molgaard et al "Thermal Conductivity of Magnetite and Hematite")
    # (assuming catalyst is Ionex - which is mainly Iron Oxide)
    # note this function has very minimal impact on results.
    def calculate_thermal_conductivity(self, temperature):
        k = 8.39 - 6.63e-3*temperature
        return k

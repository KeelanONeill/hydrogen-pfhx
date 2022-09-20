import numpy as np
from hydrogen_pfhx import heat_transfer_models, pressure_models


class PlateFinHex(object):
    def __init__(self, reactor_configuration):
        # keeping constant for now
        self.hot_fraction = 0.5

        # input
        self.length = reactor_configuration['length']
        self.width = reactor_configuration['width']
        self.height = reactor_configuration['height']
        self.fin_thickness = reactor_configuration['fin_thickness']
        self.fin_pitch = reactor_configuration['fin_pitch']  # (in width direction)
        self.fin_height = reactor_configuration['fin_height']    # (in height direction)
        self.seration_length = reactor_configuration['seration_length']

    def characteristic_length(self):
        cl = self.hydraulic_diameter
        return cl

    def calculate_fin_efficiency(self, h, k):
        I = self.fin_height/2 - self.fin_thickness
        m = (2*h/(k*self.fin_thickness) * (1 + self.fin_thickness/I))**0.5
        fe = np.tanh(m*I)/(m*I)
        return fe

    def fin_spacing(self):
        s = self.fin_pitch - self.fin_thickness
        return s

    def layer_height(self):
        lh = self.fin_height + self.fin_thickness
        return lh

    def number_layers(self):
        # ensuring even number of layers (for hot/cold)
        nl = np.floor(self.height / self.layer_height()/2)*2
        return nl

    def hydraulic_perimeter(self):
        ph = 2*(self.fin_height + self.fin_spacing())
        return ph

    def hydraulic_diameter(self):
        dh = 4*self.single_channel_area() / self.hydraulic_perimeter()
        return dh

    def total_channels(self):
        channels_per_layer = np.floor(self.width / self.fin_pitch)
        tc = channels_per_layer * self.number_layers()
        return tc

    def hot_channels(self):
        hc = self.hot_fraction * self.total_channels()
        return hc

    def cold_channels(self):
        cc = (1-self.hot_fraction)*self.total_channels()
        return cc

    def single_channel_area(self):
        csa = self.fin_height * self.fin_spacing()
        return csa

    def total_channel_area(self):
        tca = self.single_channel_area() * self.total_channels()
        return tca

    def total_cold_side_area(self):
        tcca = self.single_channel_area() * self.cold_channels()
        return tcca

    def total_hot_side_area(self):
        thca = self.single_channel_area() * self.hot_channels()
        return thca

    def calculate_heat_transfer_duty(self, reactant_temperature, coolant_temperature):
        wall_temperature = (coolant_temperature + reactant_temperature) / 2
        (U, terms) = self.calculate_overall_heat_transfer_coefficient(
            wall_temperature)
        Q = U*self.hydraulic_perimeter()*(reactant_temperature - coolant_temperature) * \
            self.hot_channels()  # W/m
        return Q, terms

    def hot_side_pressure_drop(self, reactant, catalyst):
        voidage = catalyst.void_fraction
        vs = voidage * reactant.velocity
        Dp = catalyst.particle_diameter
        rho = reactant.mass_density
        delta_p = -(150*reactant.viscosity * (1-voidage)**2 * vs / (Dp**2 *
                    voidage**3) + 1.75 * rho * (1 - voidage) * vs**2 / (Dp * voidage ** 3))
        return delta_p

    def cold_side_pressure_drop(self, coolant):
        # non-negative due to direction
        delta_p = 2 * self.f_cold * coolant.mass_density * \
            coolant.velocity**2 / (self.hydraulic_diameter())
        return delta_p

    def calculate_overall_heat_transfer_coefficient(self, wall_temperatures):
        solid_thermal_conductivity = np.mean(
            heat_transfer_models.aluminium_thermal_conductivity(wall_temperatures))
        term_1 = self.hydraulic_diameter()/(8*self.k_hot)
        term_2 = 1/(self.h_hot)
        term_3 = self.fin_thickness / solid_thermal_conductivity
        term_4 = 1/(self.h_cold)
        terms = [term_1, term_2, term_3, term_4]
        u = 1/np.sum(terms)
        return u, terms

    # friction factor (f) & heat transfer coefficient (h) for hot side.
    def hot_side_transport_behaviour(self, reactant, catalyst, wall_temperature):
        # initial transport properties
        self.k_hot = self.effective_radial_conductivity(
            reactant, catalyst, self.hydraulic_diameter())
        Pr_eff = reactant.viscosity * reactant.specific_heat_capacity / \
            reactant.thermal_conductivity  # self.k_hot

        # wall properties
        original_temperature = reactant.temperature
        reactant.update_conditions(wall_temperature, reactant.pressure)
        k_wall = self.effective_radial_conductivity(
            reactant, catalyst, self.hydraulic_diameter())
        Pr_w = reactant.viscosity * reactant.specific_heat_capacity / k_wall
        reactant.update_conditions(original_temperature, reactant.pressure)

        # Reynolds number
        reactant.calculate_reynolds_number(catalyst.particle_diameter)
        Re = reactant.reynolds_number
        aspect_ratio = self.hydraulic_diameter() / self.length

        # friction factor
        self.f_hot = pressure_models.ergun_equation(catalyst.void_fraction, Re)

        # Nusselt number correlations
        if Re < 0:
            print('Cannot have negative Reynolds')
        elif Re <= 2000:
            Nu_h = heat_transfer_models.laminar_nusselt_correlation(
                aspect_ratio)
        elif (Re > 2000) & (Re < 8000):
            f_darcy = pressure_models.ergun_equation(
                catalyst.void_fraction, Re)
            Nu_lam = heat_transfer_models.laminar_nusselt_correlation(
                aspect_ratio)

            Nu_turb = heat_transfer_models.gnielinski_equation(
                f_darcy, Re, Pr_eff, Pr_w, aspect_ratio)
            phi = 4/3 - Re/6000
            Nu_h = phi*Nu_turb + (1-phi)*Nu_lam
        else:
            f_darcy = pressure_models.ergun_equation(
                catalyst.void_fraction, Re)
            Nu_h = heat_transfer_models.gnielinski_equation(
                f_darcy, Re, Pr_eff, Pr_w, aspect_ratio)

        h_h = Nu_h*self.k_hot/self.hydraulic_diameter()

        # correct due to fin efficiency
        fe_hot = self.calculate_fin_efficiency(
            h_h, reactant.thermal_conductivity)
        self.h_hot = fe_hot*h_h


    def cold_side_transport_behaviour(self, coolant):
        # friction factor (f) & heat transfer coefficient (h) for cold side.
        alpha = self.fin_spacing() / self.fin_height
        delta = self.fin_thickness / self.seration_length
        gamma = self.fin_thickness / self.fin_spacing()
        (f, j) = heat_transfer_models.manglik_bergles_heat_transfer_model(
            coolant.reynolds_number, alpha, delta, gamma)
        self.f_cold = f
        Nu_c = j * coolant.reynolds_number * coolant.prandtl_number**(1/3)
        h_c = Nu_c*coolant.thermal_conductivity/self.hydraulic_diameter()

        # correct due to fin efficiency
        fe_cold = self.calculate_fin_efficiency(
            h_c, coolant.thermal_conductivity)
        h_c = h_c * fe_cold
        self.h_cold = h_c

    def effective_radial_conductivity(self, reactant, catalyst, d_t):
        # props
        k_f = reactant.thermal_conductivity
        k_s = catalyst.calculate_thermal_conductivity(reactant.temperature)
        epsilon = catalyst.void_fraction
        dp = catalyst.particle_diameter

        zeta = 0.9
        alpha_rs = 2e-5 * (zeta/(2-zeta))*(reactant.temperature/100)**3
        B = 1.25 * ((1 - epsilon) / epsilon)**(10/9)

        k_r = k_f/k_s
        dim_alpha = alpha_rs * dp / k_f

        omega = (1+(dim_alpha - 1)*k_r)/(1+(dim_alpha - B)*k_r) * np.log((1 + alpha_rs*dp/k_s) /
                                                                         (B*k_r)) - (B-1) / (1 + (dim_alpha - B)*k_r) + (B+1) / (2*B) * (dim_alpha - B)
        k_0 = k_f * ((1 - (1-epsilon))**0.5 * (1 + epsilon * dim_alpha) +
                     2*omega * (1-epsilon)**0.5/(1+(dim_alpha - B)*k_r))

        # dynamic thermal conductivity
        Pe = (1 + 46*(dp/d_t)**2)/0.14  # radial Peclet number
        D_er = reactant.velocity * dp / (epsilon * Pe)
        k_t = epsilon * reactant.mass_density * reactant.specific_heat_capacity * D_er

        k_eff = k_0 + k_t
        return k_eff
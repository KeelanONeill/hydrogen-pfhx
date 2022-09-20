"""
functions to model boundary value problem for PFHX
"""

import numpy as np
from hydrogen_pfhx import ortho_para_dynamics

# Main 1D BVP model of h2 conversion reactor.
def bvp_function(z, process_properties, additional_parameters):
    # extract the "additional parameters"
    # Note - must be organised in this way due to bvp_solver setup.
    reactor = additional_parameters[0]
    catalyst = additional_parameters[1]
    reactant = additional_parameters[2]
    coolant = additional_parameters[3]
    boundary_properties = additional_parameters[4]

    # (reactor, catalyst, reactant, coolant, boundary_properties)
    delta_process_properties = np.zeros(process_properties.shape)
    nodes = len(z)

    term_matrix = np.zeros((nodes, 4))
    for zi in np.arange(nodes):
        current_process_properties = process_properties[:, zi]
        # update the conditions (T & P) of the fluid streams. Also update reacting
        # stream composition.
        [reactant, coolant] = update_parameters(
            current_process_properties, reactant, coolant, boundary_properties)

        # calculate wall temperature
        wall_temperature = (coolant.temperature + reactant.temperature)/2

        # Reactant side calcs
        reactant.calculate_velocity(
            reactor.total_hot_side_area()*catalyst.void_fraction)
        reactant.calculate_reynolds_number(
            catalyst.particle_diameter/catalyst.solid_fraction)
        reactor.hot_side_transport_behaviour(
            reactant, catalyst, wall_temperature)

        # coolant side calcs
        coolant.calculate_velocity(reactor.total_cold_side_area())
        coolant.calculate_reynolds_number(reactor.hydraulic_diameter())
        reactor.cold_side_transport_behaviour(coolant)

        # reactor conversion
        r_dot = ortho_para_dynamics.first_order_kinetics(
            reactant)              # mol/m3/s
        dNpdz = r_dot * reactor.total_hot_side_area()*catalyst.solid_fraction   # mol/m/s
        dxpdz = dNpdz * reactant.molecular_mass / reactant.mass_flow_rate

        # heat transfer calculations
        (Q_transfer, _) = reactor.calculate_heat_transfer_duty(
            reactant.temperature, coolant.temperature)
        hoc = reactant.get_heat_of_conversion()
        Q_hot = -Q_transfer - hoc * dNpdz

        # negative due to direction
        dTcdz = -Q_transfer/(coolant.mass_flow_rate *
                             coolant.specific_heat_capacity)
        # Q_hot is negative as heat transferred out of reactant
        dTrdz = 1/(reactant.mass_flow_rate *
                   reactant.specific_heat_capacity)*Q_hot

        # Basic assumption that conversion in coolant is extremely slow - needs to be
        # updated.
        dPrdz = reactor.hot_side_pressure_drop(reactant, catalyst)
        dPcdz = reactor.cold_side_pressure_drop(coolant)
        current_deltas = np.vstack(
            ([dxpdz[0], dPrdz[0]/1e3, dTrdz[0], dPcdz/1e3, dTcdz[0]]))
        delta_process_properties[:, zi] = current_deltas.flatten()

    return delta_process_properties

# define boundary conditions
def counter_current_boundary_condition(inlet_properties, outlet_properties, boundary_parameters):
    # corresponds to reactant inlet (x_para, P & T)
    inlet_residual = inlet_properties[0:3] - boundary_parameters[0:3]
    # corresponds to coolant inlet (P & T)
    outlet_residual = outlet_properties[3:5] - boundary_parameters[3:5]
    bc_res = np.hstack([inlet_residual, outlet_residual])
    return bc_res

# method for defining an initial guess for the solution.
def initialise_solution(reactant, coolant, reactor, catalyst, boundary_properties, simulation_configuration):
    nodes = simulation_configuration['nodes']
    x_linear = np.linspace(0, reactor.length, nodes)
    delta_T_guess = simulation_configuration['delta_t']
    T_reactant_outlet = boundary_properties[4] + delta_T_guess
    T_coolant_outlet = boundary_properties[2] - delta_T_guess
    sol_guess = initial_guess(x_linear, boundary_properties,
                              reactor.length, T_reactant_outlet, T_coolant_outlet)
    reactant_tc = np.zeros((nodes,))
    reactant_cp = np.zeros((nodes,))
    coolant_tc = np.zeros((nodes,))
    coolant_cp = np.zeros((nodes,))
    reactant_h = np.zeros((nodes,))
    reactant_T = np.zeros((nodes,))
    coolant_h = np.zeros((nodes,))
    coolant_T = np.zeros((nodes,))

    for xi in np.arange(nodes):
        (reactant, coolant) = update_parameters(
            sol_guess[:, xi], reactant, coolant, boundary_properties)
        reactant_tc[xi] = reactant.thermal_conductivity
        reactant_cp[xi] = reactant.specific_heat_capacity
        reactant_h[xi] = reactant.enthalpy
        reactant_T[xi] = reactant.temperature
        coolant_tc[xi] = coolant.thermal_conductivity
        coolant_cp[xi] = coolant.specific_heat_capacity
        coolant_h[xi] = coolant.enthalpy
        coolant_T[xi] = coolant.temperature

    reactant_q = reactant_h * reactant.mass_flow_rate
    coolant_q = coolant_h * coolant.mass_flow_rate
    reactant_duty = reactant_q[-1] - reactant_q[0]
    coolant_duty = coolant_q[-1] - coolant_q[0]
    if abs(reactant_duty) > abs(coolant_duty):
        outlet_reactant_h = reactant_h[0] + \
            coolant_duty/reactant.mass_flow_rate
        T_reactant_outlet = np.interp(
            outlet_reactant_h, reactant_h, reactant_T)
        T_coolant_outlet = boundary_properties[2] - delta_T_guess
    elif abs(reactant_duty) < abs(coolant_duty):
        outlet_coolant_h = coolant_h[-1] - reactant_duty/coolant.mass_flow_rate
        T_coolant_outlet = np.max(
            (boundary_properties[4], np.interp(outlet_coolant_h, coolant_h, coolant_T)))
        T_reactant_outlet = boundary_properties[4] + delta_T_guess

    sol_init = initial_guess(x_linear, boundary_properties,
                             reactor.length, T_reactant_outlet, T_coolant_outlet)

    return x_linear, sol_init


def initial_guess(x, bc, length, T_reactant_outlet, T_coolant_outlet):
    nodes = len(x)
    T_reactant_inlet = bc[2]
    T_coolant_inlet = bc[4]
    T_reactant_gradient = (T_reactant_outlet - T_reactant_inlet) / length
    T_reactant_profile = T_reactant_inlet + T_reactant_gradient*x
    (final_xp, final_xo) = ortho_para_dynamics.para_ortho_equilibrium(T_reactant_outlet)
    # TODO - abstract the dxp guess into config.
    xp_adj = final_xp - 0.05
    xp_profile = bc[0] + (xp_adj - bc[0])*x/length
    T_coolant_gradient = (T_coolant_inlet - T_coolant_outlet) / length
    T_coolant_profile = T_coolant_outlet + T_coolant_gradient*x
    P_coolant = bc[3]*np.ones(nodes,)
    sol = np.concatenate([np.reshape(xp_profile, (1, nodes)),
                          np.reshape(bc[1]*(1-0.003*x / length), (1, nodes)),
                          np.reshape(T_reactant_profile, (1, nodes)),
                          np.reshape(P_coolant, (1, nodes)),
                          np.reshape(T_coolant_profile, (1, nodes))],
                         axis=0)
    return sol


def update_parameters(process_properties, reactant, coolant, boundary_properties):
    # impose a lower limit (non-zero) on all values (else functions may fail)
    nz_limit = 1e-6
    # extract the process properties
    # upper limit of 1 placed on composition.
    xp_reactant = np.max((0.0, np.min((process_properties[0], 1.0))))
    P_reactant = np.max((nz_limit, process_properties[1]))*1e3
    T_reactant = np.max((nz_limit, process_properties[2]))

    if len(process_properties) > 3:
        P_coolant = np.max((nz_limit, process_properties[3]))*1e3
        T_coolant = np.max((boundary_properties[4], process_properties[4]))
        coolant.update_conditions(T_coolant, P_coolant)
        coolant.set_properties()
    else:
        T_coolant = coolant.temperature

    # reactant can't be less than coolant
    T_reactant = np.max((T_coolant, T_reactant))

    # This will update the properties of the fluid stream (e.g. density,
    # viscosity)
    reactant.update_conditions(T_reactant, P_reactant)
    reactant.update_composition(xp_reactant)
    reactant.set_properties()

    return reactant, coolant

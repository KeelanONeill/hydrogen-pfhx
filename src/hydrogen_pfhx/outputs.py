import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from operator import sub
from hydrogen_pfhx import bvp_model


def post_process(solution, reactant, coolant, reactor, catalyst, boundary_conditions):
    z = solution.x
    xp_reactant = solution.y[0, :]
    P_reactant = solution.y[1, :]
    T_reactant = solution.y[2, :]
    P_coolant = solution.y[3, :]
    T_coolant = solution.y[4, :]

    rho_reactant = np.zeros(len(z),)
    rho_coolant = np.zeros(len(z),)
    xp_equil = np.zeros(len(z),)
    for ni in np.arange(len(z)):
        reactant, coolant = bvp_model.update_parameters(
            solution.y[:, ni], reactant, coolant, boundary_conditions)
        rho_reactant[ni] = reactant.mass_density
        rho_coolant[ni] = coolant.mass_density
        xp_equil[ni] = reactant.get_xp_equil()

    # calculate velocity
    u_reactant = reactant.mass_flow_rate / \
        (rho_reactant*reactor.total_hot_side_area()*catalyst.void_fraction)
    u_coolant = coolant.mass_flow_rate / \
        (rho_coolant*reactor.total_cold_side_area())

    columns = ('Z (m)', 'Reactant pressure (kPa)', 'Coolant pressure (kPa)', 'Reactant temperature (K)', 'Coolant temperature (K)',
               'Reactant velocity (m/s)', 'Coolant velocity (m/s)', 'Equilibrium para-hydrogen fraction (mol/mol)', 'Actual para-hydrogen fraction (mol/mol)')
    results_array = np.transpose(np.vstack(
        [z, P_reactant, P_coolant, T_reactant, T_coolant, u_reactant, u_coolant, xp_equil, xp_reactant]))
    results = pd.DataFrame(results_array, columns=columns)

    return results

def save_results(results, file_path='output/results.csv'):
    # save results
    if (file_path[:7] == 'output/') & (not os.path.isdir('output')):
        os.mkdir('output')
        
    results.to_csv(file_path)


def plot_results(results):
    # set figure fonts
    tfont = {'fontname': 'Times New Roman', 'fontweight': 'bold'}
    tfont_1 = {'fontname': 'Times New Roman'}
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['mathtext.it'] = 'Times New Roman:italic:bold'
    plt.rcParams['mathtext.cal'] = 'Times New Roman:italic'
    plt.rcParams['mathtext.default'] = 'regular'
    plt.rcParams["mathtext.fontset"] = 'custom'
    plt.rc('axes', titlesize=11)     # fontsize of the axes title
    plt.rc('axes', labelsize=11)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=9)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=9)    # fontsize of the tick labels

    labels = ('(b) Isomer conversion',
              '(a) Temperature',
              '(c) Velocity')

    ylabels = ('H$_{2,para}$ fraction (mol/mol)',
               'Temperature (K)',
               'Velocity (m/s)')

    nc = 4

    cc = [0.2, 0.3, 0.8]
    cr = [0.8, 0.3, 0.2]

    x_data = results['Z (m)']
    num_params = 3
    xlims = (0, np.max(x_data))
    l1 = 'reactant'

    nr = 1
    nc = 3
    fig, axs = plt.subplots(nr, nc, figsize=(7.2, 2.5), dpi=150)
    for p_idx in np.arange(num_params):
        if p_idx == 0:
            ci = 1
            ylims = (0, 1)
            r_data = results['Actual para-hydrogen fraction (mol/mol)']
            c_data = results['Equilibrium para-hydrogen fraction (mol/mol)']
        elif p_idx == 1:
            ci = 0
            ylims = (20, 80)
            r_data = results['Reactant temperature (K)']
            c_data = results['Coolant temperature (K)']
        elif p_idx == 2:
            ci = 2
            ylims = (0, 1.5)
            r_data = results['Reactant velocity (m/s)']
            c_data = results['Coolant velocity (m/s)']

        c1 = cr
        if p_idx == 0:
            ls = '--'
            c2 = 'k'
            l2 = 'equil.'
        else:
            ls = '-'
            c2 = cc
            l2 = 'coolant'
        curr_ax = axs[ci]
        curr_ax.plot(x_data, r_data, '-', color=c1, label=l1)
        curr_ax.plot(x_data, c_data, ls, color=c2, label=l2)

        # axis limits
        curr_ax.set_xlim(xlims)
        curr_ax.set_ylim(ylims)
        y_lim_diff = np.diff(ylims)[0]

        if p_idx == 0:
            plot_arrow(curr_ax, x_data, r_data, cr, '-|>', -1, y_lim_diff)
        elif p_idx == 1:
            plot_arrow(curr_ax, x_data, r_data, cr, '-|>', 1, y_lim_diff)
            plot_arrow(curr_ax, x_data, c_data, cc, '<|-', -1, y_lim_diff)
        elif p_idx == 2:
            plot_arrow(curr_ax, x_data, r_data, cr, '-|>', 1, y_lim_diff)
            plot_arrow(curr_ax, x_data, c_data, cc, '<|-', -1, y_lim_diff)

        # labels
        curr_ax.set_xlabel('Length along reactor (m)', **tfont)
        curr_ax.set_ylabel(ylabels[p_idx], **tfont)
        curr_ax.set_title(labels[p_idx], **tfont_1)

        # tick markers
        curr_ax.minorticks_on()
        curr_ax.tick_params(direction='in', which='minor', length=2,
                            bottom=True, top=True, left=True, right=True)
        curr_ax.tick_params(direction='in', which='major', length=4,
                            bottom=True, top=True, left=True, right=True)

        curr_ax.set_xticks(np.arange(xlims[0], xlims[1]+1, 1.0))
        if p_idx == 0:
            loc = 'lower right'
        else:
            loc = 'upper right'
        lg = curr_ax.legend(loc=loc)
        frame = lg.get_frame()
        frame.set_edgecolor('black')

    fig.tight_layout()

    fig.subplots_adjust(left=0.08, top=0.88, bottom=0.2, right=0.98)


def get_aspect(ax):
    # Total figure size
    figW, figH = ax.get_figure().get_size_inches()
    # Axis size on figure
    _, _, w, h = ax.get_position().bounds
    # Ratio of display units
    disp_ratio = (figH * h) / (figW * w)
    # Ratio of data units
    # Negative over negative because of the order of subtraction
    data_ratio = sub(*ax.get_ylim()) / sub(*ax.get_xlim())

    return disp_ratio / data_ratio


def plot_arrow(curr_ax, x_data, y_data, c, arrow, off_dir, y_lim_diff):
    cx = np.max(x_data)/2
    dx = np.max(x_data)/5
    g = np.interp(cx, x_data, np.gradient(y_data, x_data))
    cy = np.interp(cx, x_data, y_data)
    dy = dx*g

    offset = off_dir * y_lim_diff*0.06
    curr_ax.annotate('', xy=(cx+dx/2, cy+dy/2 + offset), xytext=(cx-dx/2, cy-dy/2 + offset), xycoords='data', textcoords='data',
                     arrowprops=dict(arrowstyle=arrow, color=c))

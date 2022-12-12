# simple test for verifying model
import os, numpy
import matplotlib.pyplot as plt
import model, outputs, utils

# load config from default location
# base_dir = os.getcwd()
base_dir = "C:\\Users\\00090502\\Documents\\git_repos\\fsr\\hydrogen-pfhx"
configuration = utils.load_config(base_dir + "\\tests\\default_configuration.yaml")

nodes_range = 10**(numpy.arange(-1,-5.1,-1))
data = []
for idx, nodes in enumerate(nodes_range):
    configuration['simulation']['tolerance'] = nodes

    # run the model with alt config
    print('\n###### Running Simulation ######')
    results = model.model(configuration)
    
    # plot & display results!
    outputs.plot_results(results)
    plt.show()
    
    data.append([results['Reactant temperature (K)'].array[-1], results['Actual para-hydrogen fraction (mol/mol)'].array[-1]])

    
    # save results
    # save_results_response = input('Would you like to save results [y/n]?\n')
    # if save_results_response == 'y':
    #     file_name = input('Enter a file name to for results data:\n')
    #     file_with_ext = file_name + '.csv.'
    #     outputs.save_results(results, file_with_ext)
    #     print('saved results as {}'.format(file_with_ext))
    
    print('finished')
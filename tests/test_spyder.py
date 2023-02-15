# simple test for verifying model

import matplotlib.pyplot as plt
import model, outputs, utils

# load config from default location
# base_dir = os.getcwd()
base_dir = "C:\\Users\\00090502\\Documents\\git_repos\\fsr\\hydrogen-pfhx"
configuration = utils.load_config(base_dir + "\\tests\\default_configuration.yaml")


# run the model with alt config
print('\n###### Running Simulation ######')
results = model.model(configuration)

# plot & display results!
outputs.plot_results(results)
plt.show()

data = [results['Reactant temperature (K)'].array[-1], results['Actual para-hydrogen fraction (mol/mol)'].array[-1]]

# save results
save_results_response = input('Would you like to save results [y/n]?\n')
if save_results_response == 'y':
    file_name = input('Enter a file name to for results data:\n')
    file_with_ext = file_name + '.csv.'
    outputs.save_results(results, file_with_ext)
    print('saved results as {}'.format(file_with_ext))
    
print('finished')
# simple test for verifying model
import matplotlib.pyplot as plt    
from hydrogen_pfhx import model, outputs, utils

# load config from default location
configuration = utils.load_config('tests/default_configuration.yaml')

# run the model with alt config
print('\n###### Running Simulation ######')
results = model.model(configuration)

# plot & display results!
outputs.plot_results(results)
plt.show()

# save results
save_results_response = input('Would you like to save results [y/n]?\n')
if save_results_response == 'y':
    file_name = input('Enter a file name to for results data:\n')
    file_with_ext = file_name + '.csv.'
    outputs.save_results(results, file_with_ext)
    print('saved results as {}'.format(file_with_ext))

print('finished')
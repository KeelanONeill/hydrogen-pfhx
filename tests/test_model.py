# simple test for verifying model
import matplotlib.pyplot as plt    
from hydrogen_pfhx import model, outputs, utils, configs

# load config from default location
configuration = utils.load_config(configs.default_configuration)

# run the model with alt config
results = model.model(configuration)

# plot & display results!
outputs.plot_results(results)
plt.show()

# save results
outputs.save_results(results,'output/results.csv')
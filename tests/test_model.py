# simple test for verifying model
import matplotlib.pyplot as plt    
from hydrogen_pfhx import model, outputs

# run the model with alt config
results = model.model()

# plot & display results!
outputs.plot_results(results)
plt.show()

# save results
outputs.save_results(results,'output/results.csv')
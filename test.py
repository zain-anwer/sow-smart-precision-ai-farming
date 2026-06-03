from vision.segmenter import segment_image
from vision.features import get_features, get_ndvi_proxy
from genetic_algorithm.engine import run_ga
import matplotlib.pyplot as plt
from pathlib import Path

image_path = Path(__file__).resolve().parent / "data/example_image.jpg"

cells = segment_image(image_path)
soil_quality, moisture_level = get_features(cells)
ndvi_arr = get_ndvi_proxy(cells)

individual, log, pareto_front = run_ga(soil_quality,moisture_level,ndvi_arr,20000,0.1,0.1)

print('Individual: \n',individual)

# printing labels to check kmeans zone detection


# visualizing the pareto front

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,4))

# ---------------------- CONVERGENCE CURVE ------------------------- #

generations = [info['gen'] for info in log]
best_fits = [info['best_fit'] for info in log]
avg_fits = [info['average_fit'] for info in log]

ax1.plot(generations,best_fits,label='Best Fitness',color='green',linewidth=2)
ax1.plot(generations,avg_fits,label='Average Fitness',color='blue',linestyle='--')
ax1.set_title('Convergence Graph')
ax1.set_xlabel('Generations')
ax1.set_ylabel('Fitness Scores')
ax1.grid(True)
ax1.legend()

# ---------------------- PARETO FRONT ------------------------------- #

profit_value = [idn.fitness.values[0] for idn in pareto_front]
water_deficit = [idn.fitness.values[1] for idn in pareto_front]

ax2.scatter(water_deficit,profit_value,color='red',marker='o',label='Optimal Solution')
ax2.set_title('Crop Allocation Trade-Off Matrix (Pareto Front)')
ax2.set_xlabel('Water Deficit (Minimize)')
ax2.set_ylabel('Total Profit (Maximize)')
ax2.grid(True)
ax2.legend()


plt.show()
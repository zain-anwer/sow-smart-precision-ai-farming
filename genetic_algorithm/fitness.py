import numpy as np
from math import sqrt
from data.rain_data import get_effective_rain
from data.crops import CROPS, CROP_NAMES


# this loads everything into memory once or something
# called closure I guess
def get_fitness_func(soil_quality,moisture_level,ndvi_arr,zone_labels,
                     penalty_weight = 2.0, latitude = None, longitude = None):

    def evaluate(individual):

        n = int(sqrt(len(individual)))
        individual = np.array(individual).reshape(n,n)

        total_water_req = 0
        total_profit = 0
        ndvi_bonus = 0
        soil_penalty = 0

        for i in range(n):
            for j in range(n):
                
                crop_idx = individual[i][j]
                crop_name = CROP_NAMES[crop_idx]

                sq = soil_quality[i][j]
                ml = moisture_level[i][j]
                ndvi_value = ndvi_arr[i][j]
                profit = CROPS[crop_name]['profit']
                water_req = CROPS[crop_name]['water']
                soil_min = CROPS[crop_name]['soil_min']

                total_water_req += water_req
                total_profit += profit
                ndvi_bonus += ndvi_value * profit * 0.1
                soil_gap = max(0,soil_min - sq)
                soil_penalty += soil_gap * profit * penalty_weight

        # won't be calculating the water penalty anymore
        # water_penalty = max(0,total_water_req - water_budget) * 50

        zone_penalty = compute_zone_penalty(individual,zone_labels)
        effective_rain = get_effective_rain(n*n,latitude,longitude)

        # recalculating water requirement based on annual rain pattern
        total_water_req = max(0,total_water_req - effective_rain)

        return (total_profit + ndvi_bonus - soil_penalty - zone_penalty,total_water_req)

    return evaluate

def compute_zone_penalty(grid,zone_labels,penalty=500):
    
    # calculate majority crop per zone and penalize every other cell with different crop
    
    total_penalty = 0
    num_zones = zone_labels.max() + 1
    
    for z in range(num_zones):
        
        # getting indices of zone z
        mask = (zone_labels == z)

        # using said indices to get crops per zone
        zone_crops = grid[mask]

        # bincount counts frequencies and stores them in indexes corresponding to value
        # argmax returns the index of the greatest number
        majority_crop = np.bincount(zone_crops).argmax()

        for crop in zone_crops:
            if crop != majority_crop:
                total_penalty += penalty

    return total_penalty 

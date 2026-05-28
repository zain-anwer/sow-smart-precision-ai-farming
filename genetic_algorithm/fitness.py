import numpy as np
from math import sqrt
from data.crops import CROPS, CROP_NAMES

# this loads everything into memory once or something
# called closure I guess
def fitness_function(soil_quality,moisture_level,ndvi_arr,water_budget,penalty_weight):

    def evaluate(individual):

        n = sqrt(len(individual))
    
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
                ndvi = ndvi_arr[i][j]
                profit = CROPS[crop_name]['profit']
                water_req = CROPS[crop_name]['water']
                soil_min = CROPS[crop_name]['soil_min']

                total_water_req += water_req
                total_profit += profit
                ndvi_bonus += ndvi * profit * 0.1
                soil_gap = max(0,soil_min - sq)
                soil_penalty += soil_gap * profit * penalty_weight

        water_penalty = max(0,total_water_req - water_budget) * 50
        score = soil_penalty + water_penalty + ndvi_bonus + total_profit
        return score

    return evaluate
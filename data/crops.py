# Pakistani crop data — profit (PKR/acre), water (mm/season), soil_min (0-1)
CROPS = {
"Wheat": {"profit": 45000, "water": 450, "soil_min": 0.3},
"Cotton": {"profit": 80000, "water": 700, "soil_min": 0.5},
"Rice": {"profit": 60000, "water": 1200, "soil_min": 0.6},
"Sugarcane": {"profit": 95000, "water": 1500, "soil_min": 0.7},
"Maize": {"profit": 35000, "water": 500, "soil_min": 0.4},
"Sunflower": {"profit": 40000, "water": 400, "soil_min": 0.3},
}
CROP_NAMES = list(CROPS.keys())
NUM_CROPS = len(CROP_NAMES)
import cv2
import numpy as np

# two features are extracted
# we convert BGR --> HSV (hue,saturation,value)
# hue range : [0,179]
# saturation range : [0,255]
# value range : [0,255]

def get_features(cells : list[list[np.ndarray]]):
    
    # cells -> 2D list of 3D numpy arrays
    
    r = len(cells)
    c = len(cells[0])

    moisture_level = []
    soil_quality = []

    moisture_level = np.zeros((r,c),dtype=np.float32)
    soil_quality = np.zeros((r,c),dtype=np.float32)

    for i in range(r):

        for j in range(c):
            hsv_cell = cv2.cvtColor(cells[i][j],cv2.COLOR_BGR2RGB)
            hue_channel = hsv_cell[:,:,0].astype(np.float32)

            # normalizing the saturation channel [0,1]
            saturation_channel = hsv_cell[:,:,1].astype(np.float32) / 255

            # normalizing the value channel [0,1]
            value_channel = hsv_cell[:,:,2].astype(np.float32) / 255

            # formula: mean(s) * (1 - mean(v))
            # favours higher saturation and lower brightness (dark and saturated)
            soil_quality[i,j] = np.mean(saturation_channel) * (1 - np.mean(value_channel))

            # blue and green tones
            blue_green_mask = (hue_channel > 90) & (hue_channel < 150)

            # calculating moisture as the mean of saturation on blue and green squares
            # if no such squares exist it takes it as 0.3 percent of the mean of saturation
            moisture_level[i,j] = np.mean(saturation_channel[blue_green_mask]) if blue_green_mask.any() else np.mean(saturation_channel) * 0.3

    # min-max normalizing both the arrays with a smoothing factor of 1e-8 to prevent division by zero

    soil_quality = (soil_quality - soil_quality.min()) / ((soil_quality.max() - soil_quality.min() + 1e-8))

    return soil_quality, moisture_level 

# NDVI stands for Normalized Difference Vegetation Index
# healthy plants emit more green light and absorb red light and vice versa

def get_ndvi_proxy(cells : list[list[np.ndarray]]):

    r = len(cells)
    c = len(cells[0])

    ndvi_arr = np.zeros((r,c),dtype=np.float32)

    for i in range(r):
        for j in range(c):

            cell = cells[i][j]

            green_channel = cell[:,:,1].astype(np.float32)
            red_channel = cell[:,:,2].astype(np.float32)

            numerator = green_channel - red_channel
            denominator = green_channel + red_channel + 1e-8

            ndvi_arr[i][j] = np.mean(numerator / denominator)

    # shifts the range from [-1,1] --> [0,1]
    return (ndvi_arr + 1) / 2


    
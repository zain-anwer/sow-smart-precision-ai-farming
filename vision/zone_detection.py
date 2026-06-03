import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def detect_zones(soil_quality,moisture_level,ndvi_arr,num_clusters=4):

    # since it is a 2D numpy array shape returns a tuple
    n = soil_quality.shape[0]

    # makes a 2D array of dimensions (n*n,3)
    features = np.stack(
        [
            soil_quality.flatten(),
            moisture_level.flatten(),
            ndvi_arr.flatten()
        ],
        axis=1
    )

    # since kmeans is distance based and sensitive to units and stuff we scale them
    scaled_features = StandardScaler().fit_transform(features)

    # computing labels for clusters
    kmeans = KMeans(n_clusters=num_clusters,random_state=42,n_init=10)
    zone_labels = kmeans.fit_predict(scaled_features)
    
    # since kmeans returns the labels in the form of a numpy array we can reshape it
    return zone_labels.reshape(n,n)

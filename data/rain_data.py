# MODULE NEEDS FIXING

import requests

# get lat and lon through ip geolocation
def get_effective_rain(num_cells,latitude=None,longitude=None):

    if latitude is None or longitude is None:
        return 0

    url = "https://api.open-meteo.com/v1/forecast"

    # request body
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'daily': 'precipitation_sum',
        'past_days': 365,
        'forecast_days': 0,
        'timezone': 'auto'
    }

    response = requests.get(url,params=params).json()
    daily_rain = response['daily']['precipitation_sum']

    # daily rain list may contain None values
    annual_rain = sum([x for x in daily_rain if x is not None])
    
    # returing water need fulfilled by rain
    return (annual_rain * 0.7 * num_cells)

# this file will contain the streamlit UI

import streamlit as st
from streamlit_geolocation import streamlit_geolocation # type: ignore
from genetic_algorithm.engine import run_ga
from PIL import Image

# (Import your other modules here)

st.title("🌱 SowSmart Precision AI Farming")


with st.sidebar:
    
    st.subheader('-- Project Parameters --')
    st.write("Click below to sync your farm's coordinate boundaries with local weather grids.")

    # this will ask for location permission and stuff and add the location to the variable
    location = streamlit_geolocation()

    # isolating latitude and longitude fields
    if location and location['latitude'] is not None:
        lat = location['latitude']
        lon = location['longitude']
        st.success(f"📍 GPS Coordinates Saved: Lat {lat:.4f}, Lon {lon:.4f}")
    
    else:
        st.info("Using default agricultural baseline coordinates.")
        default_lat, default_lon = 40.7128, -74.0060

    # grid size
    grid_n = st.slider('Grid Size (N)',1,20,10,1)

    # population size
    pop_size = st.slider('Population Size: ',50,300,100,1)

    # number of generations before stopping
    gen_size = st.slider('Number Of Generations: ',50,300,100,1)

    # soil penalty weight   
    penalty_w = st.slider('Soil penalty weight', 0.5, 5.0, 2.0, 0.5)

    # image to be processed
    uploaded_image = st.file_uploader(
    label = 'Upload Farm Image',
    type = ['jpeg','jpg','png']
    )

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image,caption='Uploaded Image',use_container_width=True)
        st.stop()


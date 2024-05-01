import streamlit as st
import pickle
import numpy as np


def load_m():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_m()

model_loaded = data["model"]
le_Light_Conditions = data["le_Light_Conditions"]
le_Road_Surface_Conditions = data["le_Road_Surface_Conditions"]
le_Road_Type = data["le_Road_Type"]
le_Urban_or_Rural_Area = data["le_Urban_or_Rural_Area"]
le_Weather_Conditions = data["le_Weather_Conditions"]
le_Vehicle_Type = data["le_Vehicle_Type"]

def show_predict_page():
    st.title("Accident Severity Prediction")

    st.write("""### please select from the below drop-down to predict the severity""")

    Light_Conditions = (
        "Daylight",
        "Darkness",
    )

    Road_Surface_Conditions = (
        "Dry",
        "Frost_or_Snow",
        "Wet_or_damp",
    )

    Road_Type = (
        "Single_carriageway",
        "Dual_carriageway",
        "One_way_street",
        "Roundabout",
    )

    Urban_or_Rural_Area = (
        "Urban",
        "Rural",
    )

    Weather_Conditions = (
        "Fine",
        "Raining",
        "Snowing/Fog_or_mist",
    )

    Vehicle_Type = (
        "Car",
        "Motorcycle",
        "Agricultural_or_Other_heavy_vehicle",
        "Goods_Carrier",
        "Passenger_Vehicles(Minibus_or_Bus)",
    )

    Light_Conditions = st.selectbox("Light Conditions", Light_Conditions)
    Number_of_Casualties = st.slider("Number of Casualties", 1, 10, 1)
    Number_of_Vehicles = st.slider("Number of Vehicles Collided", 1, 10, 1)
    Road_Surface_Conditions = st.selectbox("Road Surface Condition", Road_Surface_Conditions)

    Road_Type = st.selectbox("Road Type", Road_Type)
    Urban_or_Rural_Area = st.selectbox("Urban or Rural Area", Urban_or_Rural_Area)
    Weather_Conditions = st.selectbox("Weather Conditions", Weather_Conditions)
    Vehicle_Type = st.selectbox("Vehicle Type", Vehicle_Type)

    ok = st.button("Predict Severity")
    if ok:
        X = np.array([[Light_Conditions, Number_of_Casualties, Number_of_Vehicles, Road_Surface_Conditions, Road_Type, Urban_or_Rural_Area, Weather_Conditions, Vehicle_Type]])
        X[:,0] = le_Light_Conditions.transform(X[:,0])
        X[:,1] = X[:,1]
        X[:,2] = X[:,2]
        X[:,3] = le_Road_Surface_Conditions.transform(X[:,3])
        X[:,4] = le_Road_Type.transform(X[:,4])
        X[:,5] = le_Urban_or_Rural_Area.transform(X[:,5])
        X[:,6] = le_Weather_Conditions.transform(X[:,6])
        X[:,7] = le_Vehicle_Type.transform(X[:,7])

        X = X.astype(float)

        pred = model_loaded.predict(X)
        st.subheader(f"The accident severity is {pred}")
        
show_predict_page()

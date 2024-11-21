import streamlit as st
import numpy as np
import joblib
import math
from datetime import date

# Load Models and Scalers
earthquake_model = joblib.load("earthquake_rf_model.pkl")
fire_model = joblib.load("fire_risk_model.pkl")
fire_scaler = joblib.load("scaler.pkl")

# Function: Fire Risk Assessment
def assess_fire_risk(temp, hum, ws, r):
    risk_score = 0
    if temp > 35:
        risk_score += 3
    elif temp > 30:
        risk_score += 2
    elif temp > 25:
        risk_score += 1
    if hum < 30:
        risk_score += 3
    elif hum < 40:
        risk_score += 2
    elif hum < 50:
        risk_score += 1
    if ws > 10:
        risk_score += 2
    elif ws > 5:
        risk_score += 1
    if r > 10:
        risk_score = max(0, risk_score - 2)
    elif r > 5:
        risk_score = max(0, risk_score - 1)
    return min(risk_score / 10, 1.0)

# Fire Risk Index Calculation Functions
def _ffmc(temps, rhs, wss, rains, ffmc_old):
    wmo = 150.0
    if rains > 0.5:
        wmo = 147.2 * (101.0 - ffmc_old) / (59.5 + ffmc_old)
    else:
        if rains > 0:
            wmo_add = (42.5 * rains * math.exp(-100.0 / (251.0 - wmo)) * (1.0 - math.exp(-6.93 / rains))
                        + 0.0015 * (wmo - 150.0) ** 2 * math.sqrt(rains))
            wmo += wmo_add
            if wmo > 250.0:
                wmo = 250.0
    ed = 0.942 * (rhs ** 0.679) + (11.0 + math.exp((rhs - 100.0) / 10.0)) + 0.18
    ew = 0.618 * (rhs ** 0.753) + (10.0 + math.exp((rhs - 100.0) / 10.0)) + 0.18
    ev = (21.1 - temps) * (1.0 + 1.0 / math.exp(rhs * 0.115))

    wm = None
    if wmo <= ed and wmo < ew:
        z = (0.424 * (1.0 - ((100.0 - rhs) / 100.0) ** 1.7)
             + 0.0694 * math.sqrt(wss) * (1.0 - ((100.0 - rhs) / 100.0) ** 8.0))
        x = z * 0.0579 * math.exp(0.0365 * temps)
        wm = ew + (wmo - ew) * math.exp(-2.303 * x)
    elif wmo > ed:
        z = (0.424 * (1.0 - (rhs / 100.0) ** 1.7)
             + 0.0694 * math.sqrt(wss) * (1.0 - (rhs / 100.0) ** 8.0))
        x = z * 0.0579 * math.exp(0.0365 * temps)
        wm = ed + (wmo - ed) * math.exp(-2.303 * x)

    ffmcs = float(59.5 * (250.0 - wm) / (147.2 + wm))
    return min(max(ffmcs, 0.0), 101.0)

def _dmc(temps, rhs, rains, dmc_old, month):
    cffdrs_el = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5 ]
    month_cffdrs_el = cffdrs_el[month - 1]
    if rains > 0:
        return min(dmc_old + 0.3 * (temps - month_cffdrs_el) + rains * 0.3, 60.0)
    else:
        return min(dmc_old + 0.3 * (temps - month_cffdrs_el), 60.0)

def _dc(temps, rains, dc_old, month):
    cffdrs_el = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5]
    month_cffdrs_el = cffdrs_el[month - 1]
    if rains > 0:
        return min(dc_old + 0.5 * (temps - month_cffdrs_el) + rains * 0.5, 200.0)
    else:
        return min(dc_old + 0.5 * (temps - month_cffdrs_el), 200.0)

def _isi(ffmc, ws):
    return (0.1 * ffmc) * (math.sqrt(ws))

def _bui(dmc, dc):
    return 0.8 * dmc + 0.2 * dc

def _fwi(bui, isi):
    return bui * isi / 100

# App Pages
def home_page():
    st.title("Welcome to the Disaster Prediction System")
    st.write("Use this app to predict:")
    st.write("- **Earthquake likelihood**")
    st.write("- **Forest Fire risk**")
    
    option = st.selectbox(
        "Choose a model to use:",
        ("Select", "Earthquake Detection", "Forest Fire Risk Prediction")
    )

    # Button to proceed
    proceed_button = st.button("Proceed to Prediction")

    # Only show the proceed button if a model is selected
    if option != "Select" and proceed_button:
        if option == "Earthquake Detection":
            st.session_state.page = "earthquake"
        elif option == "Forest Fire Risk Prediction":
            st.session_state.page = "fire"

def earthquake_page():

    st.title("Earthquake Detection App")
   

    # Input Fields
    latitude = st.number_input("Latitude (deg)", value=0.0)
    longitude = st.number_input("Longitude (deg)", value=0.0)
    depth = st.number_input("Depth (km)", value=0.0)
    no_of_stations = st.number_input("Number of Stations", value=1)

    # Prediction Button
    if st.button("Predict"):
        # Combine inputs into an array for prediction
        input_data = np.array([latitude, longitude, depth, no_of_stations]).reshape(1, -1)
        prediction = earthquake_model.predict(input_data)
        result = "Earthquake Detected" if prediction[0] == 1 else "No Earthquake Detected"
        st.write(f"**Prediction:** {result}")

    if st.button("Go Back"):
        st.session_state.page = "home"

def fire_page():
    st.title("Fire Risk Prediction App")
    st.write("Enter the details below to assess the risk of fire for a specific day.")

    # Date Input
    date_input = st.date_input("Select a Date", min_value=date.today())

    # Weather Inputs
    temp = st.number_input("Enter Temperature (°C)", min_value=-50, max_value=50)
    hum = st.number_input("Enter Humidity (%)", min_value=0, max_value=100)
    ws = st.number_input("Enter Wind Speed (mph)", min_value=0, max_value=100)
    r = st.number_input("Enter Precipitation (inches)", min_value=0.0, max_value=10.0)

    # Fire indices calculations
    ffmc_i, dmc_i, dc_i = 85.0, 6.0, 15.0
    month = date_input.month
    ffmc = _ffmc(temp, hum, ws, r, ffmc_i)
    dmc = _dmc(temp, hum, r, dmc_i, month)
    dc = _dc(temp, r, dc_i, month)
    isi = _isi(ffmc, ws)
    bui = _bui(dmc, dc)
    fwi = _fwi(bui, isi)

    # Prepare input for model
    input_features = np.array([temp, hum, ws, r, ffmc, dmc, dc, isi, bui, fwi]).reshape(1, -1)
    input_scaled = fire_scaler.transform(input_features)

    # Get model probability
    ml_prob = fire_model.predict_proba(input_scaled)[0][1]

    # Custom risk probability
    custom_risk_prob = assess_fire_risk(temp, hum, ws, r)

    # Combine probabilities
    final_prob = 0.6 * ml_prob + 0.4 * custom_risk_prob

    # Display results
    if final_prob < 0.20:
        message = "Very low probability of fire"
    elif final_prob < 0.40:
        message = "Low probability of fire"
    elif final_prob < 0.60:
        message = "Moderate probability of fire"
    elif final_prob < 0.80:
        message = "High probability of fire"
    else:
        message = "Extremely high probability of fire"

    st.write(f"**Fire Risk Assessment for {date_input}:**")
    st.write(f"ML Model Probability: {ml_prob:.2f}")
    st.write(f"Custom Risk Probability: {custom_risk_prob:.2f}")
    st.write(f"Final Probability: {final_prob:.2f}")
    st.write(f"Risk Message: {message}")
    
    if st.button("Go Back"):
        st.session_state.page = "home"

# Navigation Logic
if "page" not in st.session_state:
    st.session_state.page = "home"

if st.session_state.page == "home":
    home_page()
elif st.session_state.page == "earthquake":
    earthquake_page()
elif st.session_state.page == "fire":
    fire_page()

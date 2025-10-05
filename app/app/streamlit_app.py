import streamlit as st

st.title("EV Predictive Maintenance Starter Kit 🚗🔧")

st.write("Welcome to my EV PDM prototype!")

# Example input
speed = st.number_input("Enter speed (km/h):", min_value=0, max_value=300, value=50)
battery = st.number_input("Enter battery level (%):", min_value=0, max_value=100, value=80)

st.write(f"Speed: {speed} km/h, Battery: {battery}%")

# Example button
if st.button("Check Anomaly"):
    st.success("No anomaly detected ✅")

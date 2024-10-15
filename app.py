import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import joblib

# Title of the app
st.title("Purlin Section Recommender")

# Load data (Adjust paths as necessary)
@st.cache_data
def load_data():
    data_path_1 = Path(__file__).parent / 'data' / 'Futura_WuSag.csv'
    data_path_2 = Path(__file__).parent / 'data' / 'DHS_WuSag.csv'
    data_manufacturer_1 = pd.read_csv(data_path_1)
    data_manufacturer_2 = pd.read_csv(data_path_2)
    return data_manufacturer_1, data_manufacturer_2

data_manufacturer_1, data_manufacturer_2 = load_data()

# Load models and accuracies
@st.cache_resource
def load_models():
    models_dir = Path(__file__).parent / 'models'
    model_1 = joblib.load(models_dir / 'model_manufacturer_1.joblib')
    model_2 = joblib.load(models_dir / 'model_manufacturer_2.joblib')
    accuracies = joblib.load(models_dir / 'accuracies.joblib')
    accuracy_1 = accuracies['accuracy_1']
    accuracy_2 = accuracies['accuracy_2']
    return model_1, model_2, accuracy_1, accuracy_2

model_1, model_2, accuracy_1, accuracy_2 = load_models()

# Refine predictions based on cost and calculate custom accuracy
def refine_accuracy(model, features, full_data):
    # This function is not needed in the app since accuracies are pre-calculated
    pass

# Display the details of a section
def display_section_details(section):
    st.write(f"Section: {section['Section Code']}")
    st.write(f"UDL: {section['UDL Capacity']}")
    st.write(f"Length: {section['Length']}")
    st.write(f"Cost: {section['Cost']}")

# Find and display predictions and optimal sections
def find_and_display_sections(udl_input, length_input):
    # Filter the data based on user inputs
    filtered_data_1 = data_manufacturer_1[
        (data_manufacturer_1['UDL Capacity'] >= udl_input) &
        (data_manufacturer_1['Length'] >= length_input)
    ].copy()

    filtered_data_2 = data_manufacturer_2[
        (data_manufacturer_2['UDL Capacity'] >= udl_input) &
        (data_manufacturer_2['Length'] >= length_input)
    ].copy()

    # If no data is available, inform the user
    if filtered_data_1.empty and filtered_data_2.empty:
        st.error("No valid sections available from either manufacturer.")
        return

    # Use the trained models to predict section codes for the filtered data
    if not filtered_data_1.empty:
        filtered_data_1['Predicted Section'] = model_1.predict(
            filtered_data_1[['UDL Capacity', 'Length']]
        )
        optimal_section_1 = filtered_data_1.loc[filtered_data_1['Cost'].idxmin()]
    else:
        optimal_section_1 = None

    if not filtered_data_2.empty:
        filtered_data_2['Predicted Section'] = model_2.predict(
            filtered_data_2[['UDL Capacity', 'Length']]
        )
        optimal_section_2 = filtered_data_2.loc[filtered_data_2['Cost'].idxmin()]
    else:
        optimal_section_2 = None

    # Display results using Streamlit columns
    col1, col2 = st.columns(2)

    with col1:
        st.header("Manufacturer 1")
        if not filtered_data_1.empty:
            # Display predicted section
            st.subheader("AI Predicted Section")
            predicted_section_1 = filtered_data_1.iloc[0]
            display_section_details(predicted_section_1)
            st.write(f"Accuracy: {accuracy_1 * 100:.2f}%")

            # Display optimal section
            st.subheader("Optimal Section")
            display_section_details(optimal_section_1)
        else:
            st.write("No valid sections available from Manufacturer 1.")

    with col2:
        st.header("Manufacturer 2")
        if not filtered_data_2.empty:
            # Display predicted section
            st.subheader("AI Predicted Section")
            predicted_section_2 = filtered_data_2.iloc[0]
            display_section_details(predicted_section_2)
            st.write(f"Accuracy: {accuracy_2 * 100:.2f}%")

            # Display optimal section
            st.subheader("Optimal Section")
            display_section_details(optimal_section_2)
        else:
            st.write("No valid sections available from Manufacturer 2.")

    # Determine and display the most cost-effective manufacturer
    if optimal_section_1 is not None and optimal_section_2 is not None:
        if optimal_section_1['Cost'] < optimal_section_2['Cost']:
            st.success("Manufacturer 1 offers the most cost-effective option.")
        else:
            st.success("Manufacturer 2 offers the most cost-effective option.")
    elif optimal_section_1 is not None:
        st.success("Only Manufacturer 1 offers a valid section.")
    elif optimal_section_2 is not None:
        st.success("Only Manufacturer 2 offers a valid section.")
    else:
        st.error("No valid sections available from either manufacturer.")

# Get user inputs
st.sidebar.header("Input Parameters")
udl_input = st.sidebar.number_input('UDL Input [kN/m] (Less than 8 is a good range)', min_value=0.0, value=0.0)
length_input = st.sidebar.number_input('Length Input [m] (Less than 20 is a good range)', min_value=0.0, value=0.0)

# Button for predicting
if st.sidebar.button('Find Optimal Sections'):
    find_and_display_sections(udl_input, length_input)
